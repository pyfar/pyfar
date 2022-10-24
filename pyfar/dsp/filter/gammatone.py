import numpy as np
import scipy.signal as sgn
from copy import deepcopy
from deepdiff import DeepDiff
import pyfar as pf


class GammatoneBands():
    """
    Generate reconstructing auditory filter bank.

    Generate a forth order reconstructing Gammatone auditory filter bank
    according to [#]_. The center frequencies of the Gammatone filters
    are calculated using the ERB scale (see :py:func:`~erb_frequencies`).

    This is a Python port of the `hohmann2002` filter bank
    contained in the Auditory Modeling Toolbox [#]_. The filter bank can
    handle single and multi channel input and allows for an almost perfect
    reconstruction of the input signal (see examples below).

    Calling ``GFB = GammatoneBands()`` constructs the filter bank. Afterwards
    the class methods ``GFB.process()`` and ``GFB.reconstruct`` can be used to
    filter and reconstruct signals. All relevant data such as the filter
    coefficients can be obtained for example through ``GFB.coefficients``. See
    below for more documentation.

    Parameters
    ----------
    freq_range : array like
        The upper and lower frequency in Hz between which the filter bank is
        constructed. Values must be larger than 0 and not exceed half the
        sampling rate.
    resolution : number
        The frequency resolution of the filter bands in equivalent rectangular
        bandwidth (ERB) units. The bands of the filter bank are distributed
        linearly on the ERB scale. The default value of ``1`` results in one
        filter band per ERB. A value of ``0.5`` would result in two filter
        bands per ERB.
    reference_frequency : number
        The frequency relative to which the filter bands are distributed. The
        default is ``1000`` Hz.
    delay : number
        The delay in seconds that the filter bank will have, i.e., the delay
        that is added to the input signal after being filtered and summed
        again. The default is ``0.004`` seconds.
    sampling_rate : number
        The sampling rate of the filter bank in Hz. The default is ``44100``
        Hz.


    Examples
    --------

    .. plot::

        >>> import pyfar as pf
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>>
        >>> # generate the filter bank object
        >>> GFB = pf.dsp.filter.GammatoneBands([0, 22050])
        >>>
        >>> # apply the filter bank to an impulse
        >>> x = pf.signals.impulse(2**13)
        >>> real, imag = GFB.process(x)
        >>> env = pf.Signal(np.abs(real.time + 1j * imag.time), 44100)
        >>>
        >>> # the output is complex:
        >>> # Real part gives band-limited Gammatone output
        >>> # Imaginary part gives the Hilbert Transform thereof
        >>> # Absolute value gives the Envelope
        >>> plt.figure()
        >>> ax = pf.plot.time(real[2], label='real part')
        >>> pf.plot.time(imag[2], label='imaginary part')
        >>> pf.plot.time(env[2], label='envelope')
        >>> plt.legend()
        >>>
        >>> # show the magnitude response of the filter bank
        >>> plt.figure()
        >>> ax = pf.plot.freq(real)
        >>> ax.set_ylim(-40, 5)
        >>>
        >>> # reconstruct the filtered impulse
        >>> # the reconstruction error can be decreased
        >>> # using the filter bank parameter 'resolution'
        >>> y = GFB.reconstruct(real, imag)
        >>> plt.figure()
        >>> ax = pf.plot.time_freq(y, label="reconstructed impulse")
        >>> ax[0].set_xlim(0, .02)
        >>> ax[1].set_ylim(-40, 5)
        >>> ax[0].legend()
        >>>
        >>> # manipulate filter output before the reconstruction
        >>> # (manipulations must be applied to the real and imaginary output)
        >>> real.time[20:25] *= .5
        >>> imag.time[20:25] *= .5
        >>>
        >>> y = GFB.reconstruct(real, imag)
        >>> plt.figure()
        >>> ax = pf.plot.time_freq(
        ...     y, label="manipulated and reconstructed and impulse")
        >>> ax[0].set_xlim(0, .02)
        >>> ax[1].set_ylim(-40, 5)
        >>> ax[0].legend()


    References
    ----------
    .. [#] V. Hohmann, 'Frequency analysis and synthesis using a gammatone
           filterbank,' Acta Acust. united Ac. 88, 433-442 (2002).
    .. [#] https://amtoolbox.org/
    """

    def __init__(self, freq_range, resolution=1, reference_frequency=1000,
                 delay=0.004, sampling_rate=44100):

        # check input (remaining checks done in erb_frequencies)
        freq_range = np.asarray(freq_range)
        if np.any(freq_range < 0) or np.any(freq_range > sampling_rate / 2):
            raise ValueError(("Values in freq_range must be between 0 Hz and "
                              "sampling_rate/2"))
        if delay <= 0:
            raise ValueError("The delay must be larger than zero")
        if resolution <= 0:
            raise ValueError("The resolution must be larger than zero")

        # store user values
        self._freq_range = freq_range
        self._resolution = resolution
        self._reference_frequency = reference_frequency
        self._delay = delay
        self._sampling_rate = sampling_rate

        # compute center frequencies
        self._frequencies = erb_frequencies(
            freq_range, resolution, reference_frequency)
        # compute filter coefficients
        self._coefficients, self._normalizations = self._get_coefficients()
        # initialize the internal filter state
        self._state = None
        # compute the filter delay, phase factor, and gains
        # (required for the re-synthesis)
        self._delays, self._phase_factors = \
            self._get_delays_and_phase_factors()
        self._gains = self._get_gains()

    def __repr__(self):
        """Nice string representation of class instances"""
        return (f"Reconstructing Gammatone filter bank with {self.n_bands} "
                f"bands between {self.freq_range[0]} and {self.freq_range[1]} "
                f"Hz spaced by {self.resolution} ERB units @ "
                f"{self.sampling_rate} Hz sampling rate")

    def __eq__(self, other):
        """Check for equality of two objects."""
        return not DeepDiff(self.__dict__, other.__dict__)

    @property
    def freq_range(self):
        """Get the frequency range of the filter bank in Hz"""
        return self._freq_range

    @property
    def resolution(self):
        """Get the resolution of the filter bank in ERB units"""
        return self._resolution

    @property
    def reference_frequency(self):
        """Get the reference frequency of the filter bank in Hz"""
        return self._reference_frequency

    @property
    def frequencies(self):
        """Get the center frequencies of the Gammatone filters in Hz"""
        return self._frequencies

    @property
    def n_bands(self):
        """Get the number of bands in the filter bank"""
        return len(self.frequencies)

    @property
    def delay(self):
        """Get the desired delay of the filter bank in seconds"""
        return self._delay

    @property
    def sampling_rate(self):
        """Get the sampling rate of the filter bank in Hz"""
        return self._sampling_rate

    @property
    def coefficients(self):
        """
        Get the filter coefficients a as in Eq. (7) in Hohmann 2002 per band.
        """
        return self._coefficients

    @property
    def normalizations(self):
        """
        Get the normalization per band described below Eq. (9) in Hohmann 2002.
        """
        return self._normalizations

    @property
    def delays(self):
        """
        Get the delays required for summing the filter bands.

        Section 4 in Hohmann 2002 describes, how the delays are calculated.
        """
        return self._delays

    @property
    def gains(self):
        """
        Get the gains required for summing the filter bands.

        Section 4 in Hohmann 2002 describes, how the gains are calculated.
        """
        return self._gains

    @property
    def phase_factors(self):
        """
        Get the phase factors required for summing the filter bands.

        Section 4 in Hohmann 2002 describes, how the factors are calculated.
        """
        return self._phase_factors

    def _get_coefficients(self):
        """
        Compute the Gammatone filter coefficients

        Returns
        -------
        coefficients : numpy array
            The filter coefficients, i.e., the a_1 coefficients. The array
            has as many coefficients as self.frequencies.
        normalizations : numpy array
            The normalization factors, i.e., the b_0 coefficients. The array
            has as many coefficients as self.frequencies.
        """

        # Eq. (13) in Hohmann 2002
        erb_aud = 24.7 + self.frequencies / 9.265

        # Eq. (14.3) in Hohmann 2002 (precomputed values for order=4)
        a_gamma = np.pi * 720 * 2**(-6) / 36
        # Eq. (14.2) in Hohmann 2002
        b = erb_aud / a_gamma
        # Eq. (14.1) in Hohmann 2002
        lam = np.exp(-2 * np.pi * b / self.sampling_rate)
        # Eq. (10) in Hohmann 2002
        beta = 2 * np.pi * self.frequencies / self.sampling_rate
        # Eq. (1) in Hohmann 2002 (these are the a_1 coefficients)
        coefficients = lam * np.exp(1j * beta)
        # normalization from Sec. 2.2 in Hohmann 2002
        # (this is the b_0 coefficient)
        normalizations = 2 * (1-np.abs(coefficients))**4

        return coefficients, normalizations

    def _get_delays_and_phase_factors(self):
        """
        Section 4 in Hohmann 2002 describes how to derive these values. This
        is a direct Python port of the corresponding function in the AMT
        toolbox `hohmann2002_process.m`.
        """

        # the delay in samples
        delay_samples = int(np.round(self.delay * self.sampling_rate))

        # apply filterbank to impulse to estimate the required values
        real, imag = self.process(pf.signals.impulse(
            delay_samples + 3, sampling_rate=self.sampling_rate))

        # compute the envelope
        ir = real.time + 1j * imag.time
        env = np.abs(ir)

        # sample at which the maximum occurs
        # (excluding last sample for a safe calculation of the slope below)
        idx_max = np.argmax(env[:, :delay_samples + 1], axis=-1)
        delays = delay_samples - idx_max

        # calculate the phase factor from the slopes
        slopes = np.array([ir[bb, idx + 1] - ir[bb, idx - 1]
                           for bb, idx in enumerate(idx_max)])

        phase_factors = 1j / (slopes / np.abs(slopes))

        return delays, phase_factors

    def _get_gains(self):
        """
        Section 4 in Hohmann 2002 describes how to derive these values. This
        is a direct Python port of the corresponding function in the AMT
        toolbox `hohmann2002_process.m`.
        """

        # positive and negative center frequencies in the z-plane
        z = np.atleast_2d(
            np.exp(2j * np.pi * self.frequencies / self.sampling_rate)).T
        z_conj = np.conjugate(z)

        # calculate transfer function at all center frequencies for all bands
        # (matrixes contain center frequencies along first dimension and
        h_pos = (1 - np.atleast_2d(self._coefficients) / z)**(-4) * \
            np.atleast_2d(self._normalizations)
        h_neg = (1 - np.atleast_2d(self._coefficients) / z_conj)**(-4) * \
            np.atleast_2d(self._normalizations)

        # apply delay and phase correction
        phase_factors = np.atleast_2d(self._phase_factors)
        delays = np.atleast_2d(self._delays)
        h_pos *= phase_factors * z**(-delays)
        h_neg *= phase_factors * np.conjugate(z)**(-delays)

        # combine positive and negative spectrum
        h = (h_pos + np.conjugate(h_neg)) / 2

        # iteratively find gains
        gains = np.ones((self.n_bands, 1))
        for ii in range(100):
            h_fin = np.matmul(h, gains)
            gains /= np.abs(h_fin)

        return gains.flatten()

    def process(self, signal, reset=True):
        """
        Filter an input signal.

        The filter output is a complex valued time signal, whose real and
        imaginary part are returned separately.

        - If the filter bank is used for analysis purposes only, the imaginary
          part is not required for further processing.
        - If the filter bank is used for analysis and re-synthesis, any further
          processing must be applied to the real and imaginary part. Any
          complex-valued operations must be applied to the complex valued
          output as a whole.

        Parameters
        ----------
        signal : Signal
            The data to be filtered
        reset : bool, optional
            If true the internal state of the filter bank is reset before the
            filters are applied. Not resetting the state can be useful for
            blockwise processing. The default is ``True``.

        Returns
        -------
        real : Signal
            The real part of the complex output signal. This represents the
            band-limited Gammatone filter output.
        imag : Signal
            The imaginary part of the complex output signal. This approximates
            the Hilbert transform of the band-limited Gammatone filter output.

        Notes
        -----
        - ``sqrt(real.time**2 + imag.time**2)`` gives the envelope of the
          Gammatone filter output.
        - If the cshape of the output signals ``real.cshape`` and
          ``imag.cshape`` generally is ``(self.n_bands, ) + signal.cshape``.
        - An exception to this occurs if ``signal.cshape`` is ``(1, )``, i.e.,
          signal is a single channel signal. In this case the cshape of the
          output signals is ``(self.n_bands)`` and `not` ``(self.n_bands, 1)``.
        """

        # check input
        if not isinstance(signal, pf.Signal):
            raise TypeError("signal must be a pyfar Signal object")
        if signal.sampling_rate != self.sampling_rate:
            raise ValueError(("The sampling rates of the signal and Gammatone"
                              " filter bank do not match"))

        # prepare multi-dimensional signals
        time_in = signal.flatten().time
        time_out = np.zeros((self.n_bands, ) + time_in.shape, dtype=complex)

        # reset or initialize the state as a list of as many zero arrays as
        # the filter bank has bands
        if reset or self._state is None:
            state = np.zeros((4, time_in.shape[0], 2), dtype=complex)
            self._state = [state for _ in range(self.n_bands)]
        elif len(self._state) != self.n_bands \
                or self._state[0].shape != (4, time_in.shape[0], 2):
            raise ValueError((
                "The shape of the signal and the internal state of the filter "
                "bank do not match. Try calling process with reset=True "
                "or with the signal that it was previously used with."
            ))

        # apply the filter band by band
        # using second order sections is faster than a manual call of
        # sgn.lfilter four time in a row
        for bb in range(self.n_bands):

            sos_section = np.tile(np.atleast_2d(
                [1, 0, 0, 1, -self._coefficients[bb], 0]),
                (4, 1))
            sos_section[3, 0] = self._normalizations[bb]
            time_out[bb], self._state[bb] = sgn.sosfilt(
                sos_section, time_in, axis=-1, zi=self._state[bb])

        # restore original channel shape
        time_out = np.reshape(time_out,
                              (self.n_bands, ) + signal.cshape + (-1, ))

        # squeeze dimension of single channel signal
        if signal.cshape == (1, ):
            time_out = np.squeeze(time_out)

        # return real and immaginary part of output as pyfar Signal objects
        real = signal.copy()
        real.time = np.real(time_out)
        imag = signal.copy()
        imag.time = np.imag(time_out)

        return real, imag

    def reconstruct(self, real, imag):
        """
        Reconstruct filter bands.

        The summation process is described in Section 4 of Hohmann 2002 and
        uses the pre-calculated delays, phase factors and gains.

        Parameters
        ----------
        real : Signal
            The real part of the filtered input signal as returned by
            :py:func:`~filter`.
        imag : Signal
            The imaginary part of the filtered input signal as returned by
            :py:func:`~filter`.

        Returns
        -------
        reconstructed : Signal
            The summed input.  ``summed.cshape`` matches the ``cshape`` or the
            original signal before it was filtered.
        """

        # prepare output
        summed = real.copy()
        time = real.time.copy() + 1j * imag.time.copy()

        # apply phase shift, delay, and gain
        for bb, (phase_factor, delay, gain) in enumerate(zip(
                self._phase_factors, self._delays, self._gains)):

            time[bb] = \
                np.real(np.roll(time[bb], delay, axis=-1) * phase_factor) * \
                gain

        # sum and squeeze first axis (the signal is already real, but the data
        # type is still complex)
        summed.time = np.sum(np.real(time), axis=0, keepdims=True)
        if len(summed.cshape) > 1:
            summed.time = np.squeeze(summed.time, axis=0)

        return summed

    def copy(self):
        """Return a copy of the audio object."""
        return deepcopy(self)

    def _encode(self):
        # get dictionary representation
        obj_dict = self.copy().__dict__
        # define required data
        keep = ["_freq_range", "_resolution", "_reference_frequency",
                "_delay", "_sampling_rate", "_state"]
        # check if all required data is contained
        for k in keep:
            if k not in obj_dict:
                raise KeyError(f"{k} is not a class variable")
        # remove obsolete data
        for k in obj_dict.copy().keys():
            if k not in keep:
                del obj_dict[k]

        return obj_dict

    @classmethod
    def _decode(cls, obj_dict):
        # initialize new clas instance
        obj = cls(obj_dict["_freq_range"], obj_dict["_resolution"],
                  obj_dict["_reference_frequency"], obj_dict["_delay"],
                  obj_dict["_sampling_rate"])
        # set internal parameters
        obj.__dict__.update(obj_dict)

        return obj


def erb_frequencies(freq_range, resolution=1, reference_frequency=1000):
    """
    Get frequencies that are linearly spaced on the ERB frequency scale.

    The human auditory system analyzes sound in auditory filters, whose band-
    width is often given as a equivalent rectangular bandwidth (ERB). The ERB
    denotes the bandwidth of a perfect rectangular band-pass that has the same
    energy as the auditory filter. The ERB frequency scale is directly
    constructed from this concept: One ERB unit is defined as the
    frequency dependent ERB of the auditory filter at a given center frequency
    (cf. [#]_, Section 3B).

    The implementation follows Eq. (16) and (17) in [#]_ and was ported from
    the auditory modeling toolbox [#]_.

    Parameters
    ----------
    freq_range : array like
        The upper and lower frequency limits in Hz between which the frequency
        vector is computed.
    resolution : number, optional
        The frequency resolution in ERB units. The default of ``1`` returns
        frequencies that are spaced by 1 ERB unit, a value of ``0.5`` would
        return frequencies that are spaced by 0.5 ERB units.
    reference_frequency : number, optional
        The reference frequency in Hz relative to which the frequency vector
        is constructed. The default is ``1000``.

    Returns
    -------
    frequencies : numpy array
        The frequencies in Hz that are linearly distributed on the ERB scale
        with a spacing given by `resolution` ERB units.

    References
    ----------
    .. [#] B. C. J. Moore, An introduction to the psychology of hearing,
           (Leiden, Boston, Brill, 2013), 6th ed.

    .. [#] V. Hohmann, “Frequency analysis and synthesis using a gammatone
           filterbank,” Acta Acust. united Ac. 88, 433-442 (2002).

    .. [#] P. L. Søndergaard, and P. Majdak, “The auditory modeling toolbox,”
           in The technology of binaural listening, edited by J. Blauert
           (Heidelberg et al., Springer, 2013) pp. 33-56.


    """

    # check input
    if not isinstance(freq_range, (list, tuple, np.ndarray)) \
            or len(freq_range) != 2:
        raise ValueError("freq_range must be an array like of length 2")
    if freq_range[0] > freq_range[1]:
        raise ValueError(("The first value of freq_range must be smaller "
                          "than the second value"))
    if resolution <= 0:
        raise ValueError("Resolution must be larger than zero")

    # convert the frequency range and reference to ERB scale
    # (Hohmann 2002, Eq. 16)
    erb_range = 9.2645 * np.sign(freq_range) * np.log(
        1 + np.abs(freq_range) * 0.00437)
    erb_ref = 9.2645 * np.sign(reference_frequency) * np.log(
        1 + np.abs(reference_frequency) * 0.00437)

    # get the referenced range
    erb_ref_range = np.array([erb_ref - erb_range[0], erb_range[1] - erb_ref])

    # construct the frequencies on the ERB scale
    n_points = np.floor(erb_ref_range / resolution).astype(int)
    erb_points = np.arange(-n_points[0], n_points[1] + 1) * resolution \
        + erb_ref

    # convert to frequencies in Hz
    frequencies = 1 / 0.00437 * np.sign(erb_points) * (
        np.exp(np.abs(erb_points) / 9.2645) - 1)

    return frequencies

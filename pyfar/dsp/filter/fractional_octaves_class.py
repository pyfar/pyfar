import warnings
import numpy as np
import scipy.signal as sgn
from copy import deepcopy
from deepdiff import DeepDiff
import pyfar as pf


class FractionalOctaveBands():
    """Create an energy or amplitude preserving fractional octave filter bank.

    The filters are designed using second order sections of Butterworth
    band-pass filters. Note that if the upper cut-off frequency of a band lies
    above the Nyquist frequency, a high-pass filter is applied instead. Due to
    differences in the design of band-pass and high-pass filters, their slopes
    differ, potentially introducing an error in the summed energy in the stop-
    band region of the respective filters.

    .. note::
        **Usage of energy-preservation:**
        This filter bank has -3 dB cut-off frequencies. For sufficiently large
        values of ``'order'``, the summed energy of the filter bank equals the
        energy of input signal, i.e., the filter bank is energy preserving
        (reconstructing). This is useful for analysis energetic properties of
        the input signal such as the room acoustic property reverberation
        time.

        **Usage of amplitude-preservation:**
        This filter bank has -6 dB cut-off frequencies. For sufficient lengths
        of ``'n_samples'``, the summed output of the filter bank equals the
        input signal, i.e., the filter bank is amplitude preserving
        (reconstructing). This is useful for analysis and synthesis
        applications such as room acoustical simulations.
        The filters have a linear phase with a delay of ``n_samples/2`` and are
        windowed with a Hanning window to suppress side lobes of the finite
        filters. The magnitude response of the filters is designed similar to [#]_
        with two exceptions:

        1. The magnitude response is designed using squared sine/cosine ramps to
           obtain -6 dB at the cut-off frequencies.
        2. The overlap between the filters is calculated between the center and
           upper cut-off frequencies and not between the center and lower cut-off
           frequencies. This enables smaller pass-bands with unity gain, which
           might be advantageous for applications that apply analysis and
           resynthesis.

    Parameters
    ----------
    num_fractions : int, optional
        The number of bands an octave is divided into. Eg., ``1`` refers to
        octave bands and ``3`` to third octave bands. The default is ``1``.
    sampling_rate : int
        The sampling rate in Hz. The default is ``None``.
    frequency_range : array, tuple, optional
        The lower and upper frequency limits. The default is
        ``frequency_range=(20, 20e3)``.
    order : int, optional
        Order of the Butterworth filter. The default is ``14``.
    preservation : string, optional
        Preservation of either ``energy`` or ``amplitude``. The default
        is ``preservation='energy'``
    overlap : float
        Band overlap of the filter slopes between 0 and 1. Smaller values yield
        wider pass-bands and steeper filter slopes. The default is ``1``.
    slope : int, optional
        Number > 0 that defines the width and steepness of the filter slopes.
        Larger values yield wider pass-bands and steeper filter slopes. The
        default is ``0``.
    n_samples : int, optional
        Length of the filter in samples. Longer filters yield more exact
        filters. The default is ``2**12``.            

    Examples
    --------
    Filter an impulse into octave bands. The summed energy of all bands equals
    the energy of the input signal.

    .. plot::

        >>> import pyfar as pf
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> # generate the data
        >>> x = pf.signals.impulse(2**17)
        >>> y = pf.dsp.filter.fractional_octave_bands(
        ...     x, 1, freq_range=(20, 8e3))
        >>> # frequency domain plot
        >>> y_sum = pf.FrequencyData(
        ...     np.sum(np.abs(y.freq)**2, 0), y.frequencies)
        >>> pf.plot.freq(y)
        >>> ax = pf.plot.freq(y_sum, color='k', log_prefix=10, linestyle='--')
        >>> ax.set_title(
        ...     "Filter bands and the sum of their squared magnitudes")
        >>> plt.tight_layout()

    """

    def __init__(
            self, 
            num_fractions, 
            sampling_rate=44100, 
            freq_range=(20.0, 20e3), 
            order=14,
            preservation="energy",
            overlap=1,
            slope=0,
            n_samples=2**12):

        # check input
        freq_range = np.asarray(freq_range)
        preservations = ["energy", "amplitude"]
        if np.any(freq_range < 0) or np.any(freq_range > sampling_rate / 2):
            raise ValueError(("Values in freq_range must be between 0 Hz and "
                              "sampling_rate/2"))
        if preservation not in preservations:
            raise ValueError(("Preservation method unknown."
                              "Use 'energy' or 'amplitude'."))

        if overlap < 0 or overlap > 1:
            raise ValueError("overlap must be between 0 and 1")

        if not isinstance(slope, int) or slope < 0:
            raise ValueError("slope must be a positive integer.")                      

        # store user values
        self.num_fractions = num_fractions
        self._sampling_rate = sampling_rate
        self._freq_range = freq_range
        self.order = order
        self.preservation = preservation
        if preservation == "amplitude":
            self.overlap = overlap
            self.slope = slope
            self.n_samples = n_samples
        

        # compute center frequencies and cutoff frequencies
        self._norm_frequencies,     \
        self._exact_frequencies,    \
        self._cutoff_frequencies    \
        = self._center_frequencies(
                num_fractions=self.num_fractions,
                frequency_range=self.freq_range,
                return_cutoff=True)
        # initialize the internal filter state
        self._state = None
        # create the actual pyfar filter-object
        if preservation == "energy":
            self._filt = self._get_energy_filt()
        else:
            self._filt = self._get_amplitude_filt()    

    def __repr__(self):
        """Nice string representation of class instances"""
        return (f"{self.num_fractions}.-octave filter bank with {self.n_bands} "
                f"bands between {self.freq_range[0]} and {self.freq_range[1]} Hz "
                f", preservation method '{self.preservation}' and "
                f"{self.sampling_rate} Hz sampling rate")

    def __eq__(self, other):
        """Check for equality of two objects."""
        return not DeepDiff(self.__dict__, other.__dict__)

    @property
    def freq_range(self):
        """Get the frequency range of the filter bank in Hz"""
        return self._freq_range

    @property
    def norm_frequencies(self):
        """Get the IEC center frequencies of the (fractional) octave filters in Hz"""
        return self._norm_frequencies

    @property
    def exact_frequencies(self):
        """Get the exact center frequencies of the (fractional) octave filters in Hz"""
        return self._exact_frequencies

    @property
    def cutoff_frequencies(self):
        """Get the cutoff frequencies of the (fractional) octave filters in Hz"""
        return self._cutoff_frequencies

    @property
    def n_bands(self):
        """Get the number of bands in the filter bank"""
        return len(self._exact_frequencies)

    @property
    def sampling_rate(self):
        """Get the sampling rate of the filter bank in Hz"""
        return self._sampling_rate

    @property
    def filt(self):
        """
        Get the pyfar filtering object
        """
        return self._filt


    def _center_frequencies(
            self,
            num_fractions=1, 
            frequency_range=(20, 20e3), 
            return_cutoff=False):
        """Return the octave center frequencies according to the IEC 61260:1:2014
        standard.

        For numbers of fractions other than ``1`` and ``3``, only the
        exact center frequencies are returned, since nominal frequencies are not
        specified by corresponding standards.

        Parameters
        ----------
        num_fractions : int, optional
            The number of bands an octave is divided into. Eg., ``1`` refers to
            octave bands and ``3`` to third octave bands. The default is ``1``.
        frequency_range : array, tuple
            The lower and upper frequency limits, the default is
            ``frequency_range=(20, 20e3)``.

        Returns
        -------
        nominal : array, float
            The nominal center frequencies in Hz specified in the standard.
            Nominal frequencies are only returned for octave bands and third octave
            bands.
        exact : array, float
            The exact center frequencies in Hz, resulting in a uniform distribution
            of frequency bands over the frequency range.
        cutoff_freq : tuple, array, float
            The lower and upper critical frequencies in Hz of the bandpass filters
            for each band as a tuple corresponding to ``(f_lower, f_upper)``.
        """
        nominal = None

        f_lims = np.asarray(frequency_range)
        if f_lims.size != 2:
            raise ValueError(
                "You need to specify a lower and upper limit frequency.")
        if f_lims[0] > f_lims[1]:
            raise ValueError(
                "The second frequency needs to be higher than the first.")

        if num_fractions in [1, 3]:
            nominal, exact = self._iec_center_frequencies(
                nominal, num_fractions)

            mask = (nominal >= f_lims[0]) & (nominal <= f_lims[1])
            nominal = nominal[mask]
            exact = exact[mask]

        else:
            exact = self._exact_center_frequencies(
                num_fractions, f_lims)

        if return_cutoff:
            octave_ratio = 10**(3/10)
            freqs_upper = exact * octave_ratio**(1/2/num_fractions)
            freqs_lower = exact * octave_ratio**(-1/2/num_fractions)
            f_crit = (freqs_lower, freqs_upper)
            return nominal, exact, f_crit
        else:
            return nominal, exact


    def _exact_center_frequencies(
            self,
            num_fractions, 
            frequency_range):
        """Calculate the center frequencies of arbitrary fractional octave bands.

        Parameters
        ----------
        num_fractions : int
            The number of fractions
        frequency_range
            The upper and lower frequency limits

        Returns
        -------
        exact : array, float
            An array containing the center frequencies of the respective fractional
            octave bands

        """
        ref_freq = 1e3
        Nmax = np.around(num_fractions*(np.log2(frequency_range[1]/ref_freq)))
        Nmin = np.around(num_fractions*(np.log2(ref_freq/frequency_range[0])))

        indices = np.arange(-Nmin, Nmax+1)
        exact = ref_freq * 2**(indices / num_fractions)

        return exact


    def _iec_center_frequencies(
            self,
            nominal, 
            num_fractions):
        """Returns the exact center frequencies for fractional octave bands
        according to the IEC 61260:1:2014 standard.
        octave ratio
        .. G = 10^{3/10}
        center frequencies
        .. f_m = f_r G^{x/b}
        .. f_m = f_e G^{(2x+1)/(2b)}
        where b is the number of octave fractions, f_r is the reference frequency
        chosen as 1000Hz and x is the index of the frequency band.

        Parameters
        ----------
        num_fractions : 1, 3
            The number of octave fractions. 1 returns octave center frequencies,
            3 returns third octave center frequencies.

        Returns
        -------
        nominal : array, float
            The nominal (rounded) center frequencies specified in the standard.
            Nominal frequencies are only returned for octave bands and third octave
            bands
        exact : array, float
            The exact center frequencies, resulting in a uniform distribution of
            frequency bands over the frequency range.
        """
        if num_fractions == 1:
            nominal = np.array([
                31.5, 63, 125, 250, 500, 1e3,
                2e3, 4e3, 8e3, 16e3], dtype=float)
        elif num_fractions == 3:
            nominal = np.array([
                25, 31.5, 40, 50, 63, 80, 100, 125, 160,
                200, 250, 315, 400, 500, 630, 800, 1000,
                1250, 1600, 2000, 2500, 3150, 4000, 5000,
                6300, 8000, 10000, 12500, 16000, 20000], dtype=float)

        reference_freq = 1e3
        octave_ratio = 10**(3/10)

        iseven = np.mod(num_fractions, 2) == 0
        if ~iseven:
            indices = np.around(
                num_fractions * np.log(nominal/reference_freq)
                / np.log(octave_ratio))
            exponent = (indices/num_fractions)
        else:
            indices = np.around(
                2.0*num_fractions *
                np.log(nominal/reference_freq) / np.log(octave_ratio) - 1)/2
            exponent = ((2*indices + 1) / num_fractions / 2)

        exact = reference_freq * octave_ratio**exponent

        return nominal, exact


    def _get_sos(self):
        """Calculate the second order section filter coefficients of a fractional
        octave band filter bank.

        Returns
        -------
        sos : array, float
            Second order section filter coefficients with shape (.., 6)

        Notes
        -----
        This function uses second order sections of butterworth filters for
        increased numeric accuracy and stability.
        """

        f_crit = self._cutoff_frequencies

        freqs_upper = f_crit[1]
        freqs_lower = f_crit[0]

        # normalize interval such that the Nyquist frequency is 1
        Wns = np.vstack((freqs_lower, freqs_upper)).T / self.sampling_rate * 2

        mask_skip = Wns[:, 0] >= 1
        if np.any(mask_skip):
            Wns = Wns[~mask_skip]
            warnings.warn("Skipping bands above the Nyquist frequency")

        num_bands = np.sum(~mask_skip)
        sos = np.zeros((num_bands, self.order, 6), np.double)

        for idx, Wn in enumerate(Wns):
            # in case the upper frequency limit is above Nyquist, use a highpass
            if Wn[-1] > 1:
                warnings.warn('The upper frequency limit {} Hz is above the \
                    Nyquist frequency. Using a highpass filter instead of a \
                    bandpass'.format(np.round(freqs_upper[idx], decimals=1)))
                Wn = Wn[0]
                btype = 'highpass'
                sos_hp = sgn.butter(self.order, Wn, btype=btype, output='sos')
                sos_coeff = pf.classes.filter._extend_sos_coefficients(
                    sos_hp, self.order)
            else:
                btype = 'bandpass'
                sos_coeff = sgn.butter(
                    self.order, Wn, btype=btype, output='sos')
            sos[idx, :, :] = sos_coeff
        return sos    


    def _get_energy_filt(self):
        """Create and/or apply an energy preserving fractional octave filter bank.

        The filters are designed using second order sections of Butterworth
        band-pass filters. Note that if the upper cut-off frequency of a band lies
        above the Nyquist frequency, a high-pass filter is applied instead. Due to
        differences in the design of band-pass and high-pass filters, their slopes
        differ, potentially introducing an error in the summed energy in the stop-
        band region of the respective filters.

        .. note::
            This filter bank has -3 dB cut-off frequencies. For sufficiently large
            values of ``'order'``, the summed energy of the filter bank equals the
            energy of input signal, i.e., the filter bank is energy preserving
            (reconstructing). This is useful for analysis energetic properties of
            the input signal such as the room acoustic property reverberation
            time. For an amplitude preserving filter bank with -6 dB cut-off
            frequencies see
            :py:func:`~pyfar.dsp.filter.reconstructing_fractional_octave_bands`.

        Parameters
        ----------
        signal : Signal, None
            The signal to be filtered. Pass ``None`` to create the filter without
            applying it.
        num_fractions : int, optional
            The number of bands an octave is divided into. Eg., ``1`` refers to
            octave bands and ``3`` to third octave bands. The default is ``1``.
        sampling_rate : None, int
            The sampling rate in Hz. Only required if signal is ``None``. The
            default is ``None``.
        frequency_range : array, tuple, optional
            The lower and upper frequency limits. The default is
            ``frequency_range=(20, 20e3)``.
        order : int, optional
            Order of the Butterworth filter. The default is ``14``.

        Returns
        -------
        signal : Signal
            The filtered signal. Only returned if ``sampling_rate = None``.
        filter : FilterSOS
            Filter object. Only returned if ``signal = None``.

        Examples
        --------
        Filter an impulse into octave bands. The summed energy of all bands equals
        the energy of the input signal.

        .. plot::

            >>> import pyfar as pf
            >>> import numpy as np
            >>> import matplotlib.pyplot as plt
            >>> # generate the data
            >>> x = pf.signals.impulse(2**17)
            >>> y = pf.dsp.filter.fractional_octave_bands(
            ...     x, 1, freq_range=(20, 8e3))
            >>> # frequency domain plot
            >>> y_sum = pf.FrequencyData(
            ...     np.sum(np.abs(y.freq)**2, 0), y.frequencies)
            >>> pf.plot.freq(y)
            >>> ax = pf.plot.freq(y_sum, color='k', log_prefix=10, linestyle='--')
            >>> ax.set_title(
            ...     "Filter bands and the sum of their squared magnitudes")
            >>> plt.tight_layout()

        """
        sos = self._get_sos()

        filt = pf.FilterSOS(sos, self.sampling_rate)
        filt.comment = (
            f"Second order section 1/{self.num_fractions} fractional octave band"
            f"filter of order {self.order}")

        return filt


    def _get_amplitude_filt(self):
        """
        Create and/or apply an amplitude preserving fractional octave filter bank.

        .. note::
            This filter bank has -6 dB cut-off frequencies. For sufficient lengths
            of ``'n_samples'``, the summed output of the filter bank equals the
            input signal, i.e., the filter bank is amplitude preserving
            (reconstructing). This is useful for analysis and synthesis
            applications such as room acoustical simulations. For an energy
            preserving filter bank with -3 dB cut-off frequencies see
            :py:func:`~pyfar.dsp.filter.fractional_octave_bands`.

        The filters have a linear phase with a delay of ``n_samples/2`` and are
        windowed with a Hanning window to suppress side lobes of the finite
        filters. The magnitude response of the filters is designed similar to [#]_
        with two exceptions:

        1. The magnitude response is designed using squared sine/cosine ramps to
        obtain -6 dB at the cut-off frequencies.
        2. The overlap between the filters is calculated between the center and
        upper cut-off frequencies and not between the center and lower cut-off
        frequencies. This enables smaller pass-bands with unity gain, which
        might be advantageous for applications that apply analysis and
        resynthesis.

        Parameters
        ----------
        signal : Signal, None
            The Signal to be filtered. Pass ``None`` to create the filter without
            applying it.
        num_fractions : int, optional
            Octave fraction, e.g., ``3`` for third-octave bands. The default is
            ``1``.
        frequency_range : tuple, optional
            Frequency range for fractional octave in Hz. The default is
            ``(63, 16000)``
        overlap : float
            Band overlap of the filter slopes between 0 and 1. Smaller values yield
            wider pass-bands and steeper filter slopes. The default is ``1``.
        slope : int, optional
            Number > 0 that defines the width and steepness of the filter slopes.
            Larger values yield wider pass-bands and steeper filter slopes. The
            default is ``0``.
        n_samples : int, optional
            Length of the filter in samples. Longer filters yield more exact
            filters. The default is ``2**12``.
        sampling_rate : int
            Sampling frequency in Hz. The default is ``None``. Only required if
            ``signal=None``.

        Returns
        -------
        signal : Signal
            The filtered signal. Only returned if ``sampling_rate = None``.
        filter : FilterFIR
            FIR Filter object. Only returned if ``signal = None``.
        frequencies : np.ndarray
            Center frequencies of the filters.

        References
        ----------
        .. [#] Antoni, J. (2010). Orthogonal-like fractional-octave-band filters.
            J. Acous. Soc. Am., 127(2), 884–895, doi: 10.1121/1.3273888

        Examples
        --------

        Filter and re-synthesize an impulse signal.

        .. plot::

            >>> import pyfar as pf
            >>> import numpy as np
            >>> import matplotlib.pyplot as plt
            >>> # generate data
            >>> x = pf.signals.impulse(2**12)
            >>> y, f = pf.dsp.filter.reconstructing_fractional_octave_bands(x)
            >>> y_sum = pf.Signal(np.sum(y.time, 0), y.sampling_rate)
            >>> # time domain plot
            >>> ax = pf.plot.time_freq(y_sum, color='k')
            >>> pf.plot.time(x, ax=ax[0])
            >>> ax[0].set_xlim(-5, 2**12/44100 * 1e3 + 5)
            >>> ax[0].set_title("Original (blue) and reconstructed pulse (black)")
            >>> # frequency domain plot
            >>> pf.plot.freq(y_sum, color='k', ax=ax[1])
            >>> pf.plot.freq(y, ax=ax[1])
            >>> ax[1].set_title(
            ...     "Reconstructed (black) and filtered impulse (colored)")
            >>> plt.tight_layout()
        """

        # number of frequency bins
        n_bins = int(self.n_samples // 2 + 1)

        # discard fractional octaves, if the center frequency exceeds
        # half the sampling rate
        f_id = self._exact_frequencies < self.sampling_rate / 2
        if not np.all(f_id):
            warnings.warn("Skipping bands above the Nyquist frequency")

        # DFT lines of the lower cut-off and center frequency as in
        # Antoni, Eq. (14)
        k_1 = np.round(self.n_samples * self._cutoff_frequencies[0][f_id]   \
            / self.sampling_rate).astype(int)
        k_m = np.round(self.n_samples * self._exact_frequencies[f_id]       \
            / self.sampling_rate).astype(int)
        k_2 = np.round(self.n_samples * self.cutoff_frequencies[1][f_id]    \
            / self.sampling_rate).astype(int)

        # overlap in samples (symmetrical around the cut-off frequencies)
        P = np.round(self.overlap / 2 * (k_2 - k_m)).astype(int)
        # initialize array for magnitude values
        g = np.ones((len(k_m), n_bins))

        # calculate the magnitude responses
        # (start at 1 to make the first fractional octave band as the low-pass)
        for b_idx in range(1, len(k_m)):

            if P[b_idx] > 0:
                # calculate phi_l for Antoni, Eq. (19)
                p = np.arange(-P[b_idx], P[b_idx] + 1)
                # initialize phi_l in the range [-1, 1]
                # (Antoni suggest to initialize this in the range of [0, 1] but
                # that yields wrong results and might be an error in the original
                # paper)
                phi = p / P[b_idx]
                # recursion if slope>0 as in Antoni, Eq. (20)
                for _ in range(self.slope):
                    phi = np.sin(np.pi / 2 * phi)
                # shift range to [0, 1]
                phi = .5 * (phi + 1)

                # apply fade out to current channel
                g[b_idx - 1, k_1[b_idx] - P[b_idx]:k_1[b_idx] + P[b_idx] + 1] = \
                    np.cos(np.pi / 2 * phi)
                # apply fade in in to next channel
                g[b_idx, k_1[b_idx] - P[b_idx]:k_1[b_idx] + P[b_idx] + 1] = \
                    np.sin(np.pi / 2 * phi)

            # set current and next channel to zero outside their range
            g[b_idx - 1, k_1[b_idx] + P[b_idx]:] = 0.
            g[b_idx, :k_1[b_idx] - P[b_idx]] = 0.

        # Force -6 dB at the cut-off frequencies. This is not part of Antony (2010)
        g = g**2

        # generate linear phase
        frequencies = pf.dsp.fft.rfftfreq(self.n_samples, self.sampling_rate)
        group_delay = self.n_samples / 2 / self.sampling_rate
        g = g.astype(complex) * np.exp(-1j * 2 * np.pi * frequencies * group_delay)

        # get impulse responses
        time = pf.dsp.fft.irfft(g, self.n_samples, self.sampling_rate, 'none')

        # window
        time *= sgn.windows.hann(time.shape[-1])

        # create filter object
        filt = pf.FilterFIR(time, self.sampling_rate)
        filt.comment = (
            "Reconstructing linear phase fractional octave filter bank."
            f"(num_fractions={self.num_fractions}, frequency_range={self.freq_range}, "
            f"overlap={self.overlap}, slope={self.slope})")


        return filt, self._exact_frequencies[f_id]

        
    def process(self, signal, reset=True):
        """
        Filter an input signal.
        """
        return self.filt.process(signal, reset)
        

    def copy(self):
        """Return a copy of the audio object."""
        return deepcopy(self)

    # def _encode(self):
    #     # get dictionary representation
    #     obj_dict = self.copy().__dict__
    #     # define required data
    #     keep = ["_freq_range", "_resolution", "_reference_frequency",
    #             "_delay", "_sampling_rate", "_state"]
    #     # check if all required data is contained
    #     for k in keep:
    #         if k not in obj_dict:
    #             raise KeyError(f"{k} is not a class variable")
    #     # remove obsolete data
    #     for k in obj_dict.copy().keys():
    #         if k not in keep:
    #             del obj_dict[k]

    #     return obj_dict

    # @classmethod
    # def _decode(cls, obj_dict):
    #     # initialize new class instance
    #     obj = cls(obj_dict["_freq_range"], obj_dict["_resolution"],
    #               obj_dict["_reference_frequency"], obj_dict["_delay"],
    #               obj_dict["_sampling_rate"])
    #     # set internal parameters
    #     obj.__dict__.update(obj_dict)

    #     return obj


"""
The following documents the audio classes and arithmethic operations for
audio data. More details and background is given in the concepts (
:py:mod:`audio classes <pyfar._concepts.audio_classes>`,
:py:mod:`Fourier transform <pyfar._concepts.fft>`,
:py:mod:`arithmetic operations <pyfar._concepts.arithmetic_operations>`).
"""

from copy import deepcopy
import warnings
import deepdiff
import numpy as np
import pyfar.dsp.fft as fft
from typing import Callable


class _Audio():
    """Abstract class for audio objects.

    This class holds all the methods and properties that are common to its
    three sub-classes :py:func:`TimeData`, :py:func:`FrequencyData`, and
    :py:func:`Signal`.
    """
    # indicate use of _Audio arithmetic operations for overloaded operators
    # (e.g. __rmul__)
    __array_priority__ = 1.0

    def __init__(self, domain, comment=None):

        # initialize valid parameter spaces
        self._VALID_DOMAINS = ["time", "freq"]

        # initialize global parameters
        self.comment = comment
        if domain in self._VALID_DOMAINS:
            self._domain = domain
        else:
            raise ValueError("Incorrect domain, needs to be time/freq.")

    def __eq__(self, other):
        """Check for equality of two objects."""
        return not deepdiff.DeepDiff(self.__dict__, other.__dict__)

    @property
    def domain(self):
        """The domain the data is stored in."""
        return self._domain

    @property
    def cshape(self):
        """
        Return channel shape.

        The channel shape gives the shape of the audio data excluding the last
        dimension, which is `n_samples` for time domain objects and
        `n_bins` for frequency domain objects.
        """
        return self._data.shape[:-1]

    def reshape(self, newshape):
        """
        Return reshaped copy of the audio object.

        Parameters
        ----------
        newshape : int, tuple
            new `cshape` of the audio object. One entry of newshape dimension
            can be ``-1``. In this case, the value is inferred from the
            remaining dimensions.

        Returns
        -------
        reshaped : Signal, FrequencyData, TimeData
            reshaped copy of the audio object.

        Notes
        -----
        The number of samples and frequency bins always remains the same.

        """

        # check input
        if not isinstance(newshape, int) and not isinstance(newshape, tuple):
            raise ValueError("newshape must be an integer or tuple")

        if isinstance(newshape, int):
            newshape = (newshape, )

        # reshape
        reshaped = deepcopy(self)
        length_last_dimension = reshaped._data.shape[-1]
        try:
            reshaped._data = reshaped._data.reshape(
                newshape + (length_last_dimension, ))
        except ValueError:
            if np.prod(newshape) != np.prod(self.cshape):
                raise ValueError((f"Can not reshape audio object of cshape "
                                  f"{self.cshape} to {newshape}"))

        return reshaped

    def flatten(self):
        """Return flattened copy of the audio object.

        Returns
        -------
        flat : Signal, FrequencyData, TimeData
            Flattened copy of audio object with
            ``flat.cshape = np.prod(audio.cshape)``

        Notes
        -----
        The number of samples and frequency bins always remains the same, e.g.,
        an audio object of ``cshape=(4,3)`` and ``n_samples=512`` will have
        ``cshape=(12, )`` and ``n_samples=512`` after flattening.

        """
        newshape = int(np.prod(self.cshape))

        return self.reshape(newshape)

    @property
    def comment(self):
        """Get comment."""
        return self._comment

    @comment.setter
    def comment(self, value):
        """Set comment."""
        self._comment = 'none' if value is None else str(value)

    def copy(self):
        """Return a copy of the audio object."""
        return deepcopy(self)

    def _return_item(self):
        raise NotImplementedError("To be implemented by derived classes.")

    def _assert_matching_meta_data(self):
        raise NotImplementedError("To be implemented by derived classes.")

    def _encode(self):
        """Return dictionary for the encoding."""
        class_dict = self.copy().__dict__
        return class_dict

    def _decode(self):
        """Return dictionary for the encoding."""
        raise NotImplementedError("To be implemented by derived classes.")

    def __getitem__(self, key):
        """
        Get copied slice of the audio object at key.

        Examples
        --------
        Get the first channel of a multi channel audio object

        >>> import pyfar as pf
        >>> signal = pf.signals.noise(10, rms=[1, 1])
        >>> first_channel = signal[0]


        """
        data = self._data[key]
        return self._return_item(data)

    def __setitem__(self, key, value):
        """
        Set channels of audio object at key.

        Examples
        --------
        Set the first channel of a multi channel audio object

        >>> import pyfar as pf
        >>> signal = pf.signals.noise(10, rms=[1, 1])
        >>> signal[0] = pf.signals.noise(10, rms=2)
        """
        self._assert_matching_meta_data(value)
        self._data[key] = value._data


class TimeData(_Audio):
    """Class for time data.

    Objects of this class contain time data which is not directly convertible
    to frequency domain, i.e., non-equidistant samples.

    """
    def __init__(self, data, times, comment=None):
        """Create TimeData object with data, and times.

        Parameters
        ----------
        data : array, double
            Raw data in the time domain. The memory layout of data is 'C'.
            E.g. data of ``shape = (3, 2, 1024)`` has 3 x 2 channels with
            1024 samples each. The data can be ``int`` or ``float`` and is
            converted to ``float`` in any case.
        times : array, double
            Times in seconds at which the data is sampled. The number of times
            must match the `size` of the last dimension of `data`.
        comment : str, optional
            A comment related to `data`. The default is ``'none'``.
        """

        _Audio.__init__(self, 'time', comment)

        self.time = data

        self._times = np.atleast_1d(np.asarray(times).flatten())
        if self._times.size != self.n_samples:
            raise ValueError(
                "The length of times must be data.shape[-1]")
        if np.any(np.diff(self._times) <= 0) and len(self._times) > 1:
            raise ValueError("Times must be monotonously increasing.")

    @property
    def time(self):
        """Return the data in the time domain."""
        return self._data

    @time.setter
    def time(self, value):
        """Set the time data."""
        # check and set the data and meta data
        data = np.atleast_2d(np.asarray(value))
        if data.dtype.kind == "i":
            data = data.astype("float")
        elif data.dtype.kind != "f":
            raise ValueError(
                f"time data is {data.dtype}  must be int or float")
        data = np.atleast_2d(np.asarray(value, dtype=float))
        self._data = data
        self._n_samples = data.shape[-1]
        # setting the domain is only required for Signal. Setting it here
        # avoids the need for overloading the setter and does not harm TimeData
        self._domain = 'time'

    @property
    def n_samples(self):
        """The number of samples."""
        return self._n_samples

    @property
    def signal_length(self):
        """The length of the data in seconds."""
        return self.times[-1]

    @property
    def times(self):
        """Time in seconds at which the signal is sampled."""
        return self._times

    def find_nearest_time(self, value):
        """Return the index that is closest to the query time.

        Parameters
        ----------
        value : float, array-like
            The times for which the indices are to be returned

        Returns
        -------
        indices : int, array-like
            The index for the given time instance. If the input was an array
            like, a numpy array of indices is returned.
        """
        times = np.atleast_1d(value)
        indices = np.zeros_like(times).astype(int)
        for idx, time in enumerate(times):
            indices[idx] = np.argmin(np.abs(self.times - time))
        return np.squeeze(indices)

    def _assert_matching_meta_data(self, other):
        """
        Check if the meta data matches across two :py:func:`TimeData` objects.
        """
        if other.__class__ != TimeData:
            raise ValueError("Comparison only valid against TimeData objects.")
        if self.n_samples != other.n_samples:
            raise ValueError("The number of samples does not match.")

    def _return_item(self, data):
        """Return new :py:func:`TimeData` object with data."""
        item = TimeData(
            data, times=self.times, comment=self.comment)
        return item

    def __repr__(self):
        """String representation of TimeData class."""
        repr_string = (
            f"TimeData:\n"
            f"{self.cshape} channels with {self.n_samples} samples")

        return repr_string

    @classmethod
    def _decode(cls, obj_dict):
        """Decode object based on its respective `_encode` counterpart."""
        obj = cls(
            obj_dict['_data'],
            obj_dict['_times'],
            obj_dict['_comment'])
        obj.__dict__.update(obj_dict)
        return obj

    def __add__(self, data):
        return add((self, data), 'time')

    def __radd__(self, data):
        return add((data, self), 'time')

    def __sub__(self, data):
        return subtract((self, data), 'time')

    def __rsub__(self, data):
        return subtract((data, self), 'time')

    def __mul__(self, data):
        return multiply((self, data), 'time')

    def __rmul__(self, data):
        return multiply((data, self), 'time')

    def __truediv__(self, data):
        return divide((self, data), 'time')

    def __rtruediv__(self, data):
        return divide((data, self), 'time')

    def __pow__(self, data):
        return power((self, data), 'time')

    def __rpow__(self, data):
        return power((data, self), 'time')

    def __matmul__(self, data):
        return matrix_multiplication(
            (self, data), 'time')

    def __rmatmul__(self, data):
        return matrix_multiplication(
            (data, self), 'time')


class FrequencyData(_Audio):
    """Class for frequency data.

    Objects of this class contain frequency data which is not directly
    convertible to the time domain, i.e., non-equidistantly spaced bins or
    incomplete spectra.

    """
    def __init__(self, data, frequencies, comment=None):
        """Create FrequencyData with data, and frequencies.

        Parameters
        ----------
        data : array, double
            Raw data in the frequency domain. The memory layout of Data is 'C'.
            E.g. data of ``shape = (3, 2, 1024)`` has 3 x 2 channels with 1024
            frequency bins each. Data can be ``int``, ``float`` or ``complex``.
            Data of type ``int`` is converted to ``float``.
        frequencies : array, double
            Frequencies of the data in Hz. The number of frequencies must match
            the size of the last dimension of data.
        comment : str, optional
            A comment related to the data. The default is ``'none'``.

        Notes
        -----
        FrequencyData objects do not support an FFT norm, because this requires
        knowledge about the sampling rate or the number of samples of the time
        signal [#]_.

        References
        ----------
        .. [#] J. Ahrens, C. Andersson, P. Höstmad, and W. Kropp, “Tutorial on
               Scaling of the Discrete Fourier Transform and the Implied
               Physical Units of the Spectra of Time-Discrete Signals,” Vienna,
               Austria, May 2020, p. e-Brief 600.

        """

        _Audio.__init__(self, 'freq', comment)

        # init
        freqs = np.atleast_1d(np.asarray(frequencies).flatten())
        self._frequencies = freqs
        self.freq = data

        # check frequencies
        if np.any(np.diff(freqs) <= 0) and len(frequencies) > 1:
            raise ValueError("Frequencies must be monotonously increasing.")
        if len(freqs) != self.n_bins:
            raise ValueError(
                "Number of frequencies does not match number of data points")

    @property
    def freq(self):
        """Return the data in the frequency domain."""
        return self._data

    @freq.setter
    def freq(self, value):
        """Set the frequency data."""

        # check data type
        data = np.atleast_2d(np.asarray(value))
        if data.dtype.kind == "i":
            data = data.astype("float")
        elif data.dtype.kind not in ["f", "c"]:
            raise ValueError((f"frequency data is {data.dtype} must be int, "
                              "float, or complex"))

        # match shape of frequencies
        if self.frequencies.size != data.shape[-1]:
            raise ValueError(
                "Number of frequency values does not match the number of"
                "frequencies.")
        self._data = data

    @property
    def frequencies(self):
        """Frequencies of the discrete signal spectrum."""
        return self._frequencies

    @property
    def n_bins(self):
        """Number of frequency bins."""
        return self._data.shape[-1]

    def find_nearest_frequency(self, value):
        """Return the index that is closest to the query frequency.

        Parameters
        ----------
        value : float, array-like
            The frequencies for which the indices are to be returned

        Returns
        -------
        indices : int, array-like
            The index for the given frequency. If the input was an array like,
            a numpy array of indices is returned.
        """
        freqs = np.atleast_1d(value)
        indices = np.zeros_like(freqs).astype(int)
        for idx, freq in enumerate(freqs):
            indices[idx] = np.argmin(np.abs(self.frequencies - freq))
        return np.squeeze(indices)

    def _assert_matching_meta_data(self, other):
        """Check if the meta data matches across two FrequencyData objects."""
        if other.__class__ != FrequencyData:
            raise ValueError(
                "Comparison only valid against FrequencyData objects.")
        if self.n_bins != other.n_bins:
            raise ValueError(
                "The number of frequency bins does not match.")

    def _return_item(self, data):
        """Return new FrequencyData object with data."""
        item = FrequencyData(
            data, frequencies=self.frequencies,
            comment=self.comment)
        return item

    def __repr__(self):
        """String representation of FrequencyData class."""
        repr_string = (
            f"FrequencyData:\n"
            f"{self.cshape} channels with {self.n_bins} frequencies\n")

        return repr_string

    @classmethod
    def _decode(cls, obj_dict):
        """Decode object based on its respective `_encode` counterpart."""
        obj = cls(
            obj_dict['_data'],
            obj_dict['_frequencies'],
            obj_dict['_comment'])
        obj.__dict__.update(obj_dict)
        return obj

    def __add__(self, data):
        return add((self, data), 'freq')

    def __radd__(self, data):
        return add((data, self), 'freq')

    def __sub__(self, data):
        return subtract((self, data), 'freq')

    def __rsub__(self, data):
        return subtract((data, self), 'freq')

    def __mul__(self, data):
        return multiply((self, data), 'freq')

    def __rmul__(self, data):
        return multiply((data, self), 'freq')

    def __truediv__(self, data):
        return divide((self, data), 'freq')

    def __rtruediv__(self, data):
        return divide((data, self), 'freq')

    def __pow__(self, data):
        return power((self, data), 'freq')

    def __rpow__(self, data):
        return power((data, self), 'freq')

    def __matmul__(self, data):
        return matrix_multiplication(
            (self, data), 'freq')

    def __rmatmul__(self, data):
        return matrix_multiplication(
            (data, self), 'freq')


class Signal(FrequencyData, TimeData):
    """Class for audio signals.

    Objects of this class contain data which is directly convertible between
    time and frequency domain (equally spaced samples and frequency bins). The
    data is always real valued in the time domain and complex valued in the
    frequency domain.

    """
    def __init__(
            self,
            data,
            sampling_rate,
            n_samples=None,
            domain='time',
            fft_norm='none',
            comment=None):
        """Create Signal with data, and sampling rate.

        Parameters
        ----------
        data : ndarray, double
            Raw data of the signal in the time or frequency domain. The memory
            layout of data is 'C'. E.g. data of ``shape = (3, 2, 1024)`` has
            3 x 2 channels with 1024 samples or frequency bins each. Time data
            is converted to ``float``. Frequency is converted to ``complex``
            and must be provided as single sided spectra, i.e., for all
            frequencies between 0 Hz and half the sampling rate.
        sampling_rate : double
            Sampling rate in Hz
        n_samples : int, optional
            Number of samples of the time signal. Required if domain is
            ``'freq'``. The default is ``None``, which assumes an even number
            of samples if the data is provided in the frequency domain.
        domain : ``'time'``, ``'freq'``, optional
            Domain of data. The default is ``'time'``
        fft_norm : str, optional
            The normalization of the Discrete Fourier Transform (DFT). Can be
            ``'none'``, ``'unitary'``, ``'amplitude'``, ``'rms'``, ``'power'``,
            or ``'psd'``. See :py:func:`~pyfar.dsp.fft.normalization` and [#]_
            for more information. The default is ``'none'``, which is typically
            used for energy signals, such as impulse responses.
        comment : str
            A comment related to `data`. The default is ``None``.

        References
        ----------
        .. [#] J. Ahrens, C. Andersson, P. Höstmad, and W. Kropp, “Tutorial on
               Scaling of the Discrete Fourier Transform and the Implied
               Physical Units of the Spectra of Time-Discrete Signals,” Vienna,
               Austria, May 2020, p. e-Brief 600.

        """

        # initialize signal specific parameters
        self._sampling_rate = sampling_rate
        self._VALID_FFT_NORMS = [
            "none", "unitary", "amplitude", "rms", "power", "psd"]

        # check fft norm
        if fft_norm in self._VALID_FFT_NORMS:
            self._fft_norm = fft_norm
        else:
            raise ValueError(("Invalid FFT normalization. Has to be "
                              f"{', '.join(self._VALID_FFT_NORMS)}, but "
                              f"found '{fft_norm}'"))
        # time / normalized frequency data (depending on domain)
        data = np.atleast_2d(data)

        # initialize domain specific parameters
        if domain == 'time':
            self._n_samples = data.shape[-1]
            times = np.atleast_1d(
                np.arange(0, self._n_samples) / sampling_rate)
            TimeData.__init__(self, data, times, comment)
        elif domain == 'freq':
            # check and set n_samples
            if n_samples is None:
                warnings.warn(
                    "Number of time samples not given, assuming an even "
                    "number of samples from the number of frequency bins.")
                n_samples = (data.shape[-1] - 1)*2
            elif n_samples > 2 * data.shape[-1] - 1:
                raise ValueError(("n_samples can not be larger than "
                                  "2 * data.shape[-1] - 2"))
            self._n_samples = n_samples
            self._n_bins = data.shape[-1]
            # Init remaining parameters
            FrequencyData.__init__(self, data, self.frequencies, comment)
            delattr(self, '_frequencies')
        else:
            raise ValueError("Invalid domain. Has to be 'time' or 'freq'.")

    @TimeData.time.getter
    def time(self):
        """Return the data in the time domain."""
        # converts the data from 'freq' to 'time'
        self.domain = 'time'

        return super().time

    @FrequencyData.freq.getter
    def freq(self):
        """Return the normalized frequency domain data.

        The normalized data is usually used for inspecting the data, e.g.,
        using plots or when extracting information such as the amplitude of
        harmonic components. Most processing operations, e.g., frequency
        domain convolution, require the non-normalized data stored as
        ``freq_raw``.
        """
        data = fft.normalization(
                self.freq_raw, self.n_samples, self.sampling_rate,
                self.fft_norm, inverse=False)
        return data

    @freq.setter
    def freq(self, value):
        """Set the normalized frequency domain data."""
        # check data type
        data = np.atleast_2d(np.asarray(value))
        if data.dtype.kind not in ["i", "f", "c"]:
            raise ValueError((f"frequency data is {data.dtype} must be int, "
                              "float, or complex"))
        # Check n_samples
        if data.shape[-1] != self.n_bins:
            warnings.warn(UserWarning((
                "Number of frequency bins changed, assuming an even "
                "number of samples from the number of frequency bins.")))
            self._n_samples = (data.shape[-1] - 1)*2
        # set domain
        self._domain = 'freq'
        # remove normalization
        data_denorm = fft.normalization(
                data, self._n_samples, self._sampling_rate,
                self._fft_norm, inverse=True)
        self._data = data_denorm.astype(complex)

    @property
    def freq_raw(self):
        """Return the frequency domain data without normalization.

        Most processing operations, e.g., frequency
        domain convolution, require the non-normalized data.
        The normalized data stored as ``freq`` is usually used for inspecting
        the data, e.g., using plots or when extracting information such as the
        amplitude of harmonic components.
        """
        self.domain = 'freq'
        return self._data

    @freq_raw.setter
    def freq_raw(self, value):
        """Set the frequency domain data without normalization."""
        data = np.atleast_2d(np.asarray(value))
        if data.dtype.kind not in ["i", "f", "c"]:
            raise ValueError((f"frequency data is {data.dtype} must be int, "
                              "float, or complex"))
        # Check n_samples
        if data.shape[-1] != self.n_bins:
            warnings.warn(UserWarning((
                "Number of frequency bins changed, assuming an even "
                "number of samples from the number of frequency bins.")))
            self._n_samples = (data.shape[-1] - 1)*2
        self._domain = 'freq'
        self._data = data.astype(complex)

    @_Audio.domain.setter
    def domain(self, new_domain):
        """Set the domain of the signal."""
        if new_domain not in self._VALID_DOMAINS:
            raise ValueError("Incorrect domain, needs to be time/freq.")

        if self._domain != new_domain:
            # Only process if we change domain
            if new_domain == 'time':
                # If the new domain should be time, we had a saved spectrum
                # (without normalization)
                # and need to do an inverse Fourier Transform
                self._data = fft.irfft(
                    self._data, self.n_samples, self._sampling_rate,
                    fft_norm='none')
            elif new_domain == 'freq':
                # If the new domain should be freq, we had sampled time data
                # and need to do a Fourier Transform (without normalization)
                self._data = fft.rfft(
                    self._data, self.n_samples, self._sampling_rate,
                    fft_norm='none')
            self._domain = new_domain

    @property
    def sampling_rate(self):
        """The sampling rate of the signal."""
        return self._sampling_rate

    @sampling_rate.setter
    def sampling_rate(self, value):
        self._sampling_rate = value

    @property
    def times(self):
        """Time instances the signal is sampled at."""
        return np.atleast_1d(np.arange(0, self.n_samples) / self.sampling_rate)

    @property
    def frequencies(self):
        """Frequencies of the discrete signal spectrum."""
        return np.atleast_1d(fft.rfftfreq(self.n_samples, self.sampling_rate))

    @property
    def n_bins(self):
        """Number of frequency bins."""
        return fft._n_bins(self.n_samples)

    @property
    def fft_norm(self):
        """
        The normalization for the Discrete Fourier Transform (DFT).

        See :py:func:`~pyfar.dsp.fft.normalization` and
        :py:mod:`FFT concepts <pyfar._concepts.fft>` for more information.
        """
        return self._fft_norm

    @fft_norm.setter
    def fft_norm(self, value):
        """
        The normalization for the Discrete Fourier Transform (DFT).

        See :py:func:`~pyfar.dsp.fft.normalization` for more information.
        """
        # check input
        if value not in self._VALID_FFT_NORMS:
            raise ValueError(("Invalid FFT normalization. Has to be "
                              f"{', '.join(self._VALID_FFT_NORMS)}, but found "
                              f"'{value}'"))
        self._fft_norm = value

    def _assert_matching_meta_data(self, other):
        """Check if the meta data matches across two Signal objects."""
        if not isinstance(other, Signal):
            raise ValueError("Comparison only valid against Signal objects.")
        if self.sampling_rate != other.sampling_rate:
            raise ValueError("The sampling rates do not match.")
        if self.n_samples != other.n_samples:
            raise ValueError("The number of samples does not match.")
        if self.fft_norm != other.fft_norm:
            raise ValueError("The FFT norms do not match.")

    def _return_item(self, data):
        """Return new Signal object with data."""
        item = Signal(data, sampling_rate=self.sampling_rate,
                      n_samples=self.n_samples, domain=self.domain,
                      fft_norm=self.fft_norm, comment=self.comment)
        return item

    @classmethod
    def _decode(cls, obj_dict):
        """Decode object based on its respective `_encode` counterpart."""
        obj = cls(
            obj_dict['_data'],
            obj_dict['_sampling_rate'],
            obj_dict['_n_samples'])
        obj.__dict__.update(obj_dict)
        return obj

    @property
    def signal_type(self):
        """
        The signal type is ``'energy'``  if the ``fft_norm = None`` and
        ``'power'`` otherwise.
        """
        stype = 'energy' if self.fft_norm == 'none' else 'power'

        return stype

    def __repr__(self):
        """String representation of Signal class."""
        repr_string = (
            f"{self.domain} domain {self.signal_type} Signal:\n"
            f"{self.cshape} channels with {self.n_samples} samples @ "
            f"{self._sampling_rate} Hz sampling rate and {self.fft_norm} "
            "FFT normalization\n")

        return repr_string

    def __len__(self):
        """Length of the object which is the number of samples stored.
        """
        return self.n_samples

    def __iter__(self):
        """Iterator for :py:func:`Signal` objects.

        Iterate across the first dimension of a :py:func:`Signal`. The actual
        iteration is handled through numpy's array iteration.

        Examples
        --------
        Iterate channels of a :py:func:`Signal`

        >>> import pyfar as pf
        >>> signal = pf.signals.impulse(2, amplitude=[1, 1, 1])
        >>> for idx, channel in enumerate(signal):
        >>>     channel.time *= idx
        >>>     signal[idx] = channel
        """
        return _SignalIterator(self._data.__iter__(), self)


class _SignalIterator(object):
    """Iterator for :py:func:`Signal`
    """
    def __init__(self, array_iterator, signal):
        self._array_iterator = array_iterator
        self._signal = signal
        self._iterated_sig = Signal(
            signal._data[..., 0, :],
            sampling_rate=signal.sampling_rate,
            n_samples=signal.n_samples,
            domain=signal.domain,
            fft_norm=signal.fft_norm)

    def __next__(self):
        if self._signal.domain == self._iterated_sig.domain:
            data = self._array_iterator.__next__()
            self._iterated_sig._data = np.atleast_2d(data)
        else:
            raise RuntimeError("domain changes during iterations break stuff!")

        return self._iterated_sig


def add(data: tuple, domain='freq'):
    """Add pyfar audio objects, array likes, and scalars.

    Pyfar audio objects are: :py:func:`Signal`, :py:func:`TimeData`, and
    :py:func:`FrequencyData`.

    Parameters
    ----------
    data : tuple of the form ``(data_1, data_2, ..., data_N)``
        Data to be added. Can contain pyfar audio objects, array likes, and
        scalars. Pyfar audio objects can not be mixed, e.g.,
        :py:func:`TimeData` and :py:func:`FrequencyData` objects do not work
        together. See below or
        :py:mod:`arithmetic operations <pyfar._concepts.arithmetic_operations>`
        for possible combinations of Signal FFT normalizations.
    domain : ``'time'``, ``'freq'``, optional
        Flag to indicate if the operation should be performed in the time or
        frequency domain. Frequency domain operations work on the raw
        spectrum (see :py:func:`pyfar.dsp.fft.normalization`). The default is
        ``'freq'``.

    Returns
    -------
    results : Signal, TimeData, FrequencyData, numpy array
        Result of the operation as numpy array, if `data` contains only array
        likes and numbers. Result as pyfar audio object if `data` contains an
        audio object.

    Notes
    -----
    The shape of arrays included in data need to match or be broadcastable
    into the ``cshape`` of the resulting audio object.

    The `fft_norm` of the result is as follows

    * If one signal has the FFT normalization ``'none'``, the results gets
      the normalization of the other signal.
    * If both signals have the same FFT normalization, the results gets the
      same normalization.
    * Other combinations raise an error.
    """
    return _arithmetic(data, domain, _add)


def subtract(data: tuple, domain='freq'):
    """Subtract pyfar audio objects, array likes, and scalars.

    Pyfar audio objects are: :py:func:`Signal`, :py:func:`TimeData`, and
    :py:func:`FrequencyData`.


    Parameters
    ----------
    data : tuple of the form (data_1, data_2, ..., data_N)
        Data to be subtracted. Can contain pyfar audio objects, array likes,
        and scalars. Pyfar audio objects can not be mixed, e.g.,
        :py:func:`TimeData` and :py:func:`FrequencyData` objects do not work
        together. See below or
        :py:mod:`arithmetic operations <pyfar._concepts.arithmetic_operations>`
        for possible combinations of Signal FFT normalizations.
    domain : ``'time'``, ``'freq'``, optional
        Flag to indicate if the operation should be performed in the time or
        frequency domain. Frequency domain operations work on the raw
        spectrum (See :py:func:`pyfar.dsp.fft.normalization`). The default is
        ``'freq'``.

    Returns
    -------
    results : Signal, TimeData, FrequencyData, numpy array
        Result of the operation as numpy array, if `data` contains only array
        likes and numbers. Result as pyfar audio object if `data` contains an
        audio object.

    Notes
    -----
    The shape of arrays included in data need to match or be broadcastable
    into the ``cshape`` of the resulting audio object.

    The `fft_norm` of the result is as follows

    * If one signal has the FFT normalization ``'none'``, the results gets
      the normalization of the other signal.
    * If both signals have the same FFT normalization, the results gets the
      same normalization.
    * Other combinations raise an error.
    """
    return _arithmetic(data, domain, _subtract)


def multiply(data: tuple, domain='freq'):
    """Multiply pyfar audio objects, array likes, and scalars.

    Pyfar audio objects are: :py:func:`Signal`, :py:func:`TimeData`, and
    :py:func:`FrequencyData`.


    Parameters
    ----------
    data : tuple of the form (data_1, data_2, ..., data_N)
        Data to be multiplied. Can contain pyfar audio objects, array likes,
        and scalars. Pyfar audio objects can not be mixed, e.g.,
        :py:func:`TimeData` and :py:func:`FrequencyData` objects do not work
        together. See below or
        :py:mod:`arithmetic operations <pyfar._concepts.arithmetic_operations>`
        for possible combinations of Signal FFT normalizations.
    domain : ``'time'``, ``'freq'``, optional
        Flag to indicate if the operation should be performed in the time or
        frequency domain. Frequency domain operations work on the raw
        spectrum (See :py:func:`pyfar.dsp.fft.normalization`). The default is
        ``'freq'``.

    Returns
    -------
    results : Signal, TimeData, FrequencyData, numpy array
        Result of the operation as numpy array, if `data` contains only array
        likes and numbers. Result as pyfar audio object if `data` contains an
        audio object.

    Notes
    -----
    The shape of arrays included in data need to match or be broadcastable
    into the ``cshape`` of the resulting audio object.

    The `fft_norm` of the result is as follows

    * If one signal has the FFT normalization ``'none'``, the results gets
      the normalization of the other signal.
    * If both signals have the same FFT normalization, the results gets the
      same normalization.
    * Other combinations raise an error.
    """
    return _arithmetic(data, domain, _multiply)


def divide(data: tuple, domain='freq'):
    """Divide pyfar audio objects, array likes, and scalars.

    Pyfar audio objects are: :py:func:`Signal`, :py:func:`TimeData`, and
    :py:func:`FrequencyData`.

    Parameters
    ----------
    data : tuple of the form (data_1, data_2, ..., data_N)
        Data to be divided. Can contain pyfar audio objects, array likes, and
        scalars. Pyfar audio objects can not be mixed, e.g.,
        :py:func:`TimeData` and :py:func:`FrequencyData` objects do not work
        together. See below or
        :py:mod:`arithmetic operations <pyfar._concepts.arithmetic_operations>`
        for possible combinations of Signal FFT normalizations.
    domain : ``'time'``, ``'freq'``, optional
        Flag to indicate if the operation should be performed in the time or
        frequency domain. Frequency domain operations work on the raw
        spectrum (See :py:func:`pyfar.dsp.fft.normalization`). The default is
        ``'freq'``.

    Returns
    -------
    results : Signal, TimeData, FrequencyData, numpy array
        Result of the operation as numpy array, if `data` contains only array
        likes and numbers. Result as pyfar audio object if `data` contains an
        audio object.

    Notes
    -----
    The shape of arrays included in data need to match or be broadcastable
    into the ``cshape`` of the resulting audio object.

    The `fft_norm` of the result is as follows

    * If the denominator signal has the FFT normalization ``'none'``, the
      result gets the normalization of the numerator signal.
    * If both signals have the same FFT normalization, the results gets the
      normalization ``'none'``.
    * Other combinations raise an error.
   """
    return _arithmetic(data, domain, _divide)


def power(data: tuple, domain='freq'):
    """Power of pyfar audio objects, array likes, and scalars.

    Pyfar audio objects are: :py:func:`Signal`, :py:func:`TimeData`, and
    :py:func:`FrequencyData`.

    Parameters
    ----------
    data : tuple of the form (data_1, data_2, ..., data_N)
        The base for which the power is calculated. Can contain pyfar audio
        objects, array likes, and scalars. Pyfar audio objects can not be
        mixed, e.g., :py:func:`TimeData` and :py:func:`FrequencyData` objects
        do not work together. See below or
        :py:mod:`arithmetic operations <pyfar._concepts.arithmetic_operations>`
        for possible combinations of Signal FFT normalizations.
    domain : ``'time'``, ``'freq'``, optional
        Flag to indicate if the operation should be performed in the time or
        frequency domain. Frequency domain operations work on the raw
        spectrum (See :py:func:`pyfar.dsp.fft.normalization`). The default is
        ``'freq'``.

    Returns
    -------
    results : Signal, TimeData, FrequencyData, numpy array
        Result of the operation as numpy array, if `data` contains only array
        likes and numbers. Result as pyfar audio object if `data` contains an
        audio object.

    Notes
    -----
    The shape of arrays included in data need to match or be broadcastable
    into the ``cshape`` of the resulting audio object.

    The `fft_norm` of the result is as follows

    * If one signal has the FFT normalization ``'none'``, the results gets
      the normalization of the other signal.
    * If both signals have the same FFT normalization, the results gets the
      same normalization.
    * Other combinations raise an error.
    """
    return _arithmetic(data, domain, _power)


def matrix_multiplication(
        data: tuple, domain='freq', axes=[(-2, -1), (-2, -1), (-2, -1)]):
    """Matrix multiplication of multidimensional pyfar audio objects and/or
    array likes.

    The multiplication is based on ``numpy.matmul`` and acts on the channels
    of audio objects (:py:func:`Signal`, :py:func:`TimeData`, and
    :py:func:`FrequencyData`). Alternatively, the ``@`` operator can be used
    for frequency domain matrix multiplications with the default parameters.

    Parameters
    ----------
    data : tuple of the form (data_1, data_2, ..., data_N)
        Data to be multiplied. Can contain pyfar audio objects and array likes.
        If multiple audio objects are passed they must be of the same type and
        their FFT normalizations must allow the multiplication (see
        :py:mod:`arithmetic operations <pyfar._concepts.arithmetic_operations>`
        and notes below).
        If audio objects and arrays are included, the arrays' shape need
        to match the audio objects' cshape (not the shape of the underlying
        time or frequency data). More Information on the requirements regarding
        the shapes and cshapes and their handling is given in the notes below.
    domain : ``'time'``, ``'freq'``, optional
        Flag to indicate if the operation should be performed in the time or
        frequency domain. Frequency domain operations work on the raw
        spectrum (see :py:func:`pyfar.dsp.fft.normalization`). The default is
        ``'freq'``.
    axes : list of 3 tuples
        Each tuple in the list specifies two axes to define the matrices for
        multiplication. The default ``[(-2, -1), (-2, -1), (-2, -1)]`` uses the
        last two axes of the input to define the matrices (first and
        second tuple) and writes the result to the last two axes of the output
        data (third tuple).

        In case of pyfar audio objects, the indices refer to the channel
        dimensions and ignore the last dimension of the underlying data that
        contains the samples or frequency bins (see
        :py:mod:`audio classes <pyfar._concepts.audio_classes>` for more
        information). For example, a signal with 4 times 2 channels and 120
        frequency bins has a cshape of ``(4, 2)``, while the shape of the
        underlying frequency data is  ``(4, 2, 120)``. The default tuple
        ``(-2, -1)`` would result in 120 matrices of shape ``(4, 2)`` used
        for the multiplication and not 4 matrices of shape ``(2, 120)``.

        If `data` contains more than two operands, the scheme given by `axes`
        refers to all of the sequential multiplications.
        See notes and examples for more details.

    Returns
    -------
    results : Signal, TimeData, FrequencyData, numpy array
        Result of the operation as numpy array, if `data` contains only array
        likes and numbers. Result as pyfar audio object if `data` contains an
        audio object.

    Notes
    -----
    Matrix muliplitcation of arrays including a time of frequency dependent
    dimension is possible by first converting these audio objects
    (:py:func:`Signal`, :py:func:`TimeData`, :py:func:`FrequencyData`).
    See example below.

    Audio objects with a one dimensional cshape are expanded to allow matrix
    multiplication:

    * If the first signal is 1-D, it is expanded to 2-D by prepending a
      dimension. For example a cshape of ``(10,)`` becomes ``(1, 10)``.
    * If the second signal is 1-D, it is expanded to 2-D by appending a
      dimension. For example a cshape of ``(10,)`` becomes ``(10, 1)``

    The shapes of array likes and cshapes of audio objects must be
    `broadcastable <https://numpy.org/doc/stable/user/basics.broadcasting.
    html>`_ except for the axes specified by the `axes` parameter.

    The `fft_norm` of the result is as follows

    * If one signal has the FFT normalization ``'none'``, the results gets
      the normalization of the other signal.
    * If both signals have the same FFT normalization, the results gets the
      same normalization.
    * Other combinations raise an error.

    Examples
    --------
    Matrix multiplication of two-dimensional signals.

    >>> a = pf.signals.impulse(10, amplitude=np.ones((2, 3)))
    >>> b = pf.signals.impulse(10, amplitude=np.ones((3, 4)))
    >>> a.cshape
    (2, 3)
    >>> b.cshape
    (3, 4)
    >>> pf.matrix_multiplication((a, b)).cshape
    (2, 4)
    >>> (a @ b).cshape
    (2, 4)

    Matrix multiplication of signal with a frequency dependent matrix
    requires to convert the matrix into a Signal object first.

    >>> x = pf.signals.impulse(10, amplitude=np.ones((3, 1)))
    >>> M = np.ones((2, 3, x.n_bins)) * x.frequencies
    >>> # convert to Signal
    >>> Ms = pf.Signal(M, x.sampling_rate, domain='freq')
    >>> # pf.matrix_multiplication((M, x)) raises an error
    >>> pf.matrix_multiplication((Ms, x)).cshape
    (2, 1)

    Matrix multiplication of three-dimensional signals. The third dimension
    needs to match or it is broadcasted (per default this refers to axis 0).

    >>> a_match = pf.signals.impulse(10, amplitude=np.ones((2, 3, 4)))
    >>> b = pf.signals.impulse(10, amplitude=np.ones((2, 4, 5)))
    >>> pf.matrix_multiplication((a_match, b)).cshape
    (2, 3, 5)
    >>> a_bcast1 = pf.signals.impulse(10, amplitude=np.ones((1, 3, 4)))
    >>> pf.matrix_multiplication((a_bcast1, b)).cshape
    (2, 3, 5)
    >>> a_bcast2 = pf.signals.impulse(10, amplitude=np.ones((3, 4)))
    >>> pf.matrix_multiplication((a_bcast2, b)).cshape
    (2, 3, 5)

    Use the `axes` parameter to multiply along first two channel dimensions.

    >>> a = pf.signals.impulse(10, amplitude=np.ones((2, 3, 4)))
    >>> b = pf.signals.impulse(10, amplitude=np.ones((3, 5, 4)))
    >>> pf.matrix_multiplication((a, b), axes=[(0, 1), (0, 1), (0, 1)]).cshape
    (2, 5, 4)

    Matrix multiplications of numpy arrays and signals.

    >>> B = np.ones((3, 4))
    >>> s1 = pf.signals.impulse(10, amplitude=np.ones((4)))
    >>> s2 = pf.signals.impulse(10, amplitude=np.ones((4, 2)))
    >>> s3 = pf.signals.impulse(10, amplitude=np.ones((2, 4)))
    >>> pf.matrix_multiplication((B, s1)).cshape
    (3, 1)
    >>> pf.matrix_multiplication((B, s2)).cshape
    (3, 2)
    >>> pf.matrix_multiplication(
    >>>     (B, s3), axes=[(-2, -1), (-1, -2), (-1, -2)]).cshape
    (2, 3)

    Fancy use of the `axes` parameter.

    >>> a = pf.signals.impulse(10, amplitude=np.ones((2, 3, 4)))
    >>> b = pf.signals.impulse(10, amplitude=np.ones((4, 3, 6)))
    >>> pf.matrix_multiplication((a, b), axes=[(0, 1), (1, 2), (2, 1)]).cshape
    (4, 6, 2)

    Extension of a signal with a 1-D ``cshape``.

    >>> a = pf.signals.impulse(10, amplitude=np.ones((2,)))
    >>> a.cshape
    (2,)
    >>> b = pf.signals.impulse(10, amplitude=np.ones((3, 2, 4)))
    >>> pf.matrix_multiplication((a, b)).cshape
    (3, 1, 4)
    >>> a = pf.signals.impulse(10, amplitude=np.ones((2, 3, 4)))
    >>> b = pf.signals.impulse(10, amplitude=np.ones((4, )))
    >>> pf.matrix_multiplication((a, b)).cshape
    (2, 3, 1)

    """
    return _arithmetic(data, domain, _matrix_multiplication, axes=axes)


def _arithmetic(data: tuple, domain: str, operation: Callable, **kwargs):
    """Apply arithmetic operations."""

    # check input and obtain meta data of new signal
    division = True if operation == _divide else False
    matmul = True if operation == _matrix_multiplication else False
    sampling_rate, n_samples, fft_norm, times, frequencies, audio_type, \
        cshape = \
        _assert_match_for_arithmetic(data, domain, division, matmul)

    # apply arithmetic operation
    result = _get_arithmetic_data(data[0], domain, cshape, matmul, audio_type)

    for d in range(1, len(data)):
        if matmul:
            kwargs['audio_type'] = audio_type
        result = operation(
            result,
            _get_arithmetic_data(data[d], domain, cshape, matmul, audio_type),
            **kwargs)

    # check if to return an audio object
    if audio_type == Signal:
        # Set unnormalized spectrum
        result = Signal(
            result, sampling_rate, n_samples, domain, fft_norm='none')
        # Set fft norm
        result.fft_norm = fft_norm
    elif audio_type == TimeData:
        result = TimeData(result, times)
    elif audio_type == FrequencyData:
        result = FrequencyData(result, frequencies)

    return result


def _assert_match_for_arithmetic(data: tuple, domain: str, division: bool,
                                 matmul: bool):
    """Check if type and meta data of input is fine for arithmetic operations.

    Check if sampling rate and number of samples agree if multiple signals are
    provided. Check if arrays are numeric. Check if a power signal is contained
    in the input. Extract cshape of result.

    Input:
    data : tuple
        Can contain Signal, TimeData, FrequencyData, and array like data
    domain : str
        Domain in which the arithmetic operation should be performed. 'time' or
        'freq'.
    division : bool
        ``True`` if a division is performed, ``False`` otherwise
    matmul: bool
        ``True`` if a  matrix multiplication is performed, ``False`` otherwise

    Returns
    -------
    sampling_rate : number, None
        Sampling rate of the signals. None, if no signal is contained in `data`
    n_samples : number, None
        Number of samples of the signals. None, if no signal is contained in
        `data`
    fft_norm : str, None
        FFT norm of the first signal in `data`, if all FFT norms are None.
        Otherwise the first FFT norm that is not None is taken.
    times : numpy array, None
        The times if a TimeData object was passed. None otherwise.
    frequencies : numpy array, None
        The frequencies if a FrequencyData object was passed. None otherwise.
    audio_type : type, None
        Type of the audio class if contained in data. Otherwise None.
    cshape : tuple, None
        Largest channel shape of the audio classes if contained in data.
        Otherwise empty tuple.

    """

    # we need at least two signals
    if not isinstance(data, tuple):
        raise ValueError("Input argument 'data' must be a tuple.")

    # check validity of domain
    if domain not in ['time', 'freq']:
        raise ValueError(f"domain must be time or freq but is {domain}.")

    # properties that must match
    sampling_rate = None
    n_samples = None
    fft_norm = 'none'
    times = None
    frequencies = None
    audio_type = type(None)
    cshape = ()

    # check input types and meta data
    found_audio_data = False
    for n, d in enumerate(data):
        if isinstance(d, (Signal, TimeData, FrequencyData)):
            # store meta data upon first appearance
            if not found_audio_data:
                if isinstance(d, Signal):
                    sampling_rate = d.sampling_rate
                    n_samples = d.n_samples
                    # if a signal comes first (n==0) its fft_norm is taken
                    # directly. If a signal does not come first, (n>0, e.g.
                    # 1/signal), the fft norm is matched
                    fft_norm = d.fft_norm if n == 0 else \
                        _match_fft_norm(fft_norm, d.fft_norm, division)
                elif isinstance(d, TimeData):
                    if domain != "time":
                        raise ValueError("The domain must be 'time'.")
                    times = d.times
                elif isinstance(d, FrequencyData):
                    if domain != "freq":
                        raise ValueError("The domain must be 'freq'.")
                    frequencies = d.frequencies
                if not matmul:
                    cshape = d.cshape
                found_audio_data = True
                audio_type = type(d)

            # check if type and meta data matches after first appearance
            else:
                if not isinstance(d, audio_type):
                    raise ValueError("The audio objects do not match.")
                if isinstance(d, Signal):
                    if sampling_rate != d.sampling_rate:
                        raise ValueError("The sampling rates do not match.")
                    if n_samples != d.n_samples:
                        raise ValueError(
                            "The number of samples does not match.")
                    fft_norm = _match_fft_norm(fft_norm, d.fft_norm, division)
                elif isinstance(d, TimeData):
                    if not np.allclose(times, d.times, atol=1e-15):
                        raise ValueError(
                            "The times does not match.")
                elif isinstance(d, FrequencyData):
                    if not np.allclose(
                            frequencies, d.frequencies, atol=1e-15):
                        raise ValueError(
                            "The frequencies do not match.")
                if not matmul:
                    try:
                        cshape = np.broadcast_shapes(cshape, d.cshape)
                    except ValueError:
                        raise ValueError(
                            "The cshapes do not match.")

        # check type of non signal input
        else:
            if np.asarray(d).dtype.kind not in ["i", "f", "c"]:
                raise ValueError(
                    "Input must be of type Signal, int, float, or complex")
            if np.asarray(d).dtype.kind == "c" and domain == 'time':
                raise ValueError(
                    "Complex input can not be applied in the time domain.")

    return (sampling_rate, n_samples, fft_norm, times, frequencies, audio_type,
            cshape)


def _get_arithmetic_data(data, domain, cshape, matmul, audio_type):
    """
    Return data in desired domain without any fft normalization.

    Parameters
    ----------
    data : Signal, array like, number
        Input data
    domain : 'time', 'freq'
        Domain in which the data is returned
    cshape : tuple
        Desired channel shape of output (required for operations including
        array likes and Audio objects).
    matmul: bool
        ``True`` if a  matrix multiplication is performed, ``False`` otherwise
    audio_type : type, None
        Type of the audio class of the operation's result.

    Returns
    -------
    data_out : numpy array
        Data in desired domain without any fft normalization if data is a
        Signal. `np.asarray(data)` otherwise.
    """
    if isinstance(data, (Signal, TimeData, FrequencyData)):
        # get signal in correct domain
        if domain == "time":
            data_out = data.time.copy()
        elif domain == "freq":
            if isinstance(data, Signal):
                data_out = data.freq_raw.copy()
            else:
                data_out = data.freq.copy()
        else:
            raise ValueError(
                f"domain must be 'time' or 'freq' but found {domain}")
    else:
        data_out = np.asarray(data)
        if data_out.ndim <= len(cshape) or\
                (matmul and not isinstance(None, audio_type)):
            # extend by time/frequency axis, scalars are extended to shape (1,)
            data_out = data_out[..., None]
        elif cshape != ():
            # operation includes arrays and audio objects
            raise ValueError(
                "array dimension is larger than the channel dimensions")
    return data_out


def _add(a, b):
    return a + b


def _subtract(a, b):
    return a - b


def _multiply(a, b):
    return a * b


def _divide(a, b):
    return a / b


def _power(a, b):
    return a**b


def _matrix_multiplication(a, b, axes, audio_type):
    if not isinstance(None, audio_type):
        # adjust data and axes if output is a pyfar audio object
        # add dimension if operand is only 2d
        a = np.expand_dims(a, 0) if a.ndim == 2 else a
        b = np.expand_dims(b, 1) if b.ndim == 2 else b
        # note: axes is implicitly copied
        axes = [tuple([ax-1 if ax < 0 else ax for ax in t]) for t in axes]
    return np.matmul(a, b, axes=axes)


def _match_fft_norm(fft_norm_1, fft_norm_2, division=False):
    """
    Helper function to determine the fft_norm resulting from an
    arithmetic operation of two audio objects.

    For addition, subtraction and multiplication:
    Either: one signal has fft_norm ``'none'`` , the results gets the other
    norm.
    Or: both have the same fft_norm, the results gets the same norm.
    Other combinations raise an error.

    For division:
    Either: the denominator (fft_norm_2) is ``'none'``, the result gets the
    fft_norm of the numerator (fft_norm_1).
    Or: both have the same fft_norm, the results gets the fft_norm ``'none'``.
    Other combinations raise an error.

    Parameters
    ----------
    fft_norm_1 : str, ``'none'``, ``'unitary'``, ``'amplitude'``, ``'rms'``,
    ``'power'`` or ``'psd'``
        First fft_norm for matching.
    fft_norm_2 : str, ``'none'``, ``'unitary'``, ``'amplitude'``, ``'rms'``,
    ``'power'`` or ``'psd'``
        Second fft_norm for matching.
    division : bool
        ``False`` if arithmetic operation is addition, subtraction or
        multiplication;
        ``True`` if arithmetic operation is division.

    Returns
    -------
    fft_norm_result : str, ``'none'``, ``'unitary'``, ``'amplitude'``,
    ``'rms'``, ``'power'`` or ``'psd'``
        The fft_norm resulting from arithmetic operation.
    """

    # check if fft_norms are valid
    valid_fft_norms = ['none', 'unitary', 'amplitude', 'rms', 'power', 'psd']
    if fft_norm_1 not in valid_fft_norms:
        raise ValueError((f"fft_norm_1 is {fft_norm_1} but must be in "
                          f"{', '.join(valid_fft_norms)}"))
    if fft_norm_2 not in valid_fft_norms:
        raise ValueError((f"fft_norm_2 is {fft_norm_2} but must be in "
                          f"{', '.join(valid_fft_norms)}"))

    # check if parameter division is type bool
    if not isinstance(division, bool):
        raise TypeError("Parameter division must be type bool.")

    if not division:

        if fft_norm_1 == fft_norm_2:
            fft_norm_result = fft_norm_1

        elif fft_norm_1 == 'none':
            fft_norm_result = fft_norm_2

        elif fft_norm_2 == 'none':
            fft_norm_result = fft_norm_1

        else:
            raise ValueError(("Either one fft_norm has to be 'none' or both "
                              "fft_norms must be the same, but they are ",
                              f"{fft_norm_1} and {fft_norm_2}."))

    else:

        if fft_norm_2 == 'none':
            fft_norm_result = fft_norm_1

        elif fft_norm_1 == fft_norm_2:
            fft_norm_result = 'none'

        else:
            raise ValueError(("Either fft_norm_2 (denominator) has to be "
                              "'none' or both fft_norms must be the same, but "
                              f"they are {fft_norm_1} and {fft_norm_2}."))

    return fft_norm_result

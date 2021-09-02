"""
Container classes and arithmethic operations for audio data.

The classes :py:func:`TimeData` and :py:func:`FrequencyData` are intended to
store incomplete or non-equidistant audio data in the time and frequency
domain. The class :py:func:`Signal` can be used to store equidistant and
complete audio data that can be converted between the time and frequency
domain by means of the Fourier transform.

Arithmetic operations can be applied in the time and frequency domain and
are implemented in the methods ``add``, ``subtract``, ``multiply``, ``divide``,
and ``power``. For example, two :py:func:`Signal`, :py:func:`TimeData`, or
:py:func:`FrequencyData` instances can be added in the time domain by

>>> result = pyfar.classes.audio.add((signal_1, signal_2), 'time')

and in the frequency domain by

>>> result = pyfar.classes.audio.add((signal_1, signal_2), 'freq')

This also works with more than two instances and supports array likes and
scalar values, e.g.,

>>> result = pyfar.classes.audio.add((signal_1, 1), 'time')

In this case the scalar `1` is broadcasted, i.e., it is is added to every
sample of `signal` (or every bin in case of a frequency domain operation).

The operators ``+``, ``-``, ``*``, ``/``, and ``**`` are overloaded for
convenience. Note, however, that their behavior depends on the Audio object.
Frequency domain operations are applied for :py:func:`Signal` and
:py:func:`FrequencyData` objects, i.e,

>>> result = signal1 + signal2

is equivalent to

>>> result = pyfar.classes.audio.add((signal1, signal2), 'freq')

Time domain operations are applied for :py:func:`TimeData` objects, i.e.,

>>> result = time_data_1 + time_data_2

is equivalent to

>>> result = pyfar.classes.audio.add((time_data_1, time_data_2), 'time')

In addition to the arithmetic operations, the equality operator is overloaded
to allow comparisons

>>> signal_1 == signal_2

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

    def __init__(self, domain, comment=None, dtype=np.double):

        # initialize valid parameter spaces
        # NOTE: Some are note needed by TimeData but would have to be defined
        #       in FrequencyData and Signal otherwise.
        self._VALID_DOMAINS = ["time", "freq"]
        self._VALID_FFT_NORMS = [
            "none", "unitary", "amplitude", "rms", "power", "psd"]

        # initialize global parameters
        self.comment = comment
        self._dtype = dtype
        if domain in self._VALID_DOMAINS:
            self._domain = domain
        else:
            raise ValueError("Incorrect domain, needs to be time/freq.")

        # initialize data with nan (this should make clear that this is an
        # abstract private class that does not hold data and prevent class
        # methods to fail that require data.)
        self._data = np.atleast_2d(np.nan)

    def __eq__(self, other):
        """Check for equality of two objects."""
        return not deepdiff.DeepDiff(self.__dict__, other.__dict__)

    @property
    def domain(self):
        """The domain the data is stored in."""
        return self._domain

    @property
    def dtype(self):
        """The data type of the audio object. This can be any data type and
        precision supported by numpy."""
        return self._dtype

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
        return self.copy().__dict__

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
        if isinstance(key, (int, slice, tuple)):
            try:
                data = self._data[key]
            except KeyError:
                raise KeyError("Index is out of bounds")
        else:
            raise TypeError(
                    "Index must be int, not {}".format(type(key).__name__))

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
        if isinstance(key, (int, slice)):
            try:
                self._data[key] = value._data
            except KeyError:
                raise KeyError("Index is out of bound")
        else:
            raise TypeError(
                    "Index must be int, not {}".format(type(key).__name__))


class TimeData(_Audio):
    """Class for time data.

    Objects of this class contain time data which is not directly convertable
    to frequency domain, i.e., non-equidistant samples.

    """
    def __init__(self, data, times, comment=None, dtype=np.double):
        """Create TimeData object with data, and times.

        Parameters
        ----------
        data : array, double
            Raw data in the time domain. The memory layout of data is 'C'.
            E.g. data of ``shape = (3, 2, 1024)`` has 3 x 2 channels with
            1024 samples each.
        times : array, double
            Times in seconds at which the data is sampled. The number of times
            must match the `size` of the last dimension of `data`.
        comment : str, optional
            A comment related to `data`. The default is ``'none'``.
        dtype : string, optional
            Raw data type of the audio object. The default is `float64`.
        """

        _Audio.__init__(self, 'time', comment, dtype)

        # init data and meta data
        self._data = np.atleast_2d(np.asarray(data, dtype=dtype))

        self._n_samples = self._data.shape[-1]

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
        data = np.atleast_2d(np.asarray(value))
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
        if not isinstance(other, TimeData):
            raise ValueError("Comparison only valid against TimeData objects.")
        if self.n_samples != other.n_samples:
            raise ValueError("The number of samples does not match.")

    def _return_item(self, data):
        """Return new :py:func:`TimeData` object with data."""
        item = TimeData(
            data, times=self.times, comment=self.comment, dtype=self.dtype)
        return item

    @classmethod
    def _decode(cls, obj_dict):
        """Decode object based on its respective `_encode` counterpart."""
        obj = cls(
            obj_dict['_data'],
            obj_dict['_times'],
            obj_dict['_comment'],
            obj_dict['_dtype'])
        obj.__dict__.update(obj_dict)
        return obj

    def __add__(self, data):
        return add((self, data), 'time')

    def __sub__(self, data):
        return subtract((self, data), 'time')

    def __mul__(self, data):
        return multiply((self, data), 'time')

    def __truediv__(self, data):
        return divide((self, data), 'time')

    def __pow__(self, data):
        return power((self, data), 'time')


class FrequencyData(_Audio):
    """Class for frequency data.

    Objects of this class contain frequency data which is not directly
    convertable to the time domain, i.e., non-equidistantly spaced bins or
    incomplete spectra.

    """
    def __init__(self, data, frequencies, fft_norm=None, comment=None,
                 dtype=complex):
        """Create FrequencyData with data, and frequencies.

        Parameters
        ----------
        data : array, double
            Raw data in the frequency domain. The memory layout of Data is 'C'.
            E.g. data of ``shape = (3, 2, 1024)`` has 3 x 2 channels with 1024
            frequency bins each.
        frequencies : array, double
            Frequencies of the data in Hz. The number of frequencies must match
            the size of the last dimension of data.
        fft_norm : str, optional
            The normalization of the Discrete Fourier Transform (DFT). Can be
            ``'none'``, ``'unitary'``, ``'amplitude'``, ``'rms'``, ``'power'``,
            or ``'psd'``. See :py:func:`~pyfar.dsp.fft.normalization` and [#]_
            for more information. The default is ``'none'``, which is typically
            used for energy signals, such as impulse responses.
        comment : str, optional
            A comment related to the data. The default is ``'none'``.
        dtype : string, optional
            Raw data type of the audio object. The default is `float64`.

        References
        ----------
        .. [#] J. Ahrens, C. Andersson, P. Höstmad, and W. Kropp, “Tutorial on
               Scaling of the Discrete Fourier Transform and the Implied
               Physical Units of the Spectra of Time-Discrete Signals,” Vienna,
               Austria, May 2020, p. e-Brief 600.

        """

        _Audio.__init__(self, 'freq', comment, dtype)

        # init data
        self._data = np.atleast_2d(np.asarray(data, dtype=complex))

        # init frequencies
        self._frequencies = np.atleast_1d(np.asarray(frequencies).flatten())
        if self._frequencies.size != self.n_bins:
            raise ValueError(
                "The number of frequencies must be data.shape[-1]")
        if np.any(np.diff(self._frequencies) <= 0) and \
                len(self._frequencies) > 1:
            raise ValueError("Frequencies must be monotonously increasing.")

        if fft_norm is None:
            fft_norm = 'none'
        if fft_norm in self._VALID_FFT_NORMS:
            self._fft_norm = fft_norm
        else:
            raise ValueError(("Invalid FFT normalization. Has to be "
                              f"{', '.join(self._VALID_FFT_NORMS)}, but found "
                              f"'{fft_norm}'"))

    @property
    def freq(self):
        """Return the data in the frequency domain."""
        return self._data

    @freq.setter
    def freq(self, value):
        """Set the frequency data."""
        self._data = np.atleast_2d(np.atleast_2d(value))
        # setting the domain is only required for Signal. Setting it here
        # avoids the need for overloading the setter and does not harm
        # FrequencyData
        self._domain = 'freq'

    @property
    def frequencies(self):
        """Frequencies of the discrete signal spectrum."""
        return self._frequencies

    @property
    def n_bins(self):
        """Number of frequency bins."""
        return self._data.shape[-1]

    @property
    def fft_norm(self):
        """
        The normalization for the Discrete Fourier Transform (DFT).

        See :py:func:`~pyfar.dsp.fft.normalization` for more information.
        """
        return self._fft_norm

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
        if not isinstance(other, FrequencyData):
            raise ValueError(
                "Comparison only valid against FrequencyData objects.")
        if self.n_bins != other.n_bins:
            raise ValueError(
                "The number of frequency bins does not match.")
        if self.fft_norm != other.fft_norm:
            raise ValueError("The FFT norms do not match.")

    def _return_item(self, data):
        """Return new FrequencyData object with data."""
        item = FrequencyData(
            data, frequencies=self.frequencies, fft_norm=self.fft_norm,
            comment=self.comment, dtype=self.dtype)
        return item

    @classmethod
    def _decode(cls, obj_dict):
        """Decode object based on its respective `_encode` counterpart."""
        obj = cls(
            obj_dict['_data'],
            obj_dict['_frequencies'],
            obj_dict['_fft_norm'],
            obj_dict['_comment'],
            obj_dict['_dtype'])
        obj.__dict__.update(obj_dict)
        return obj

    def __add__(self, data):
        return add((self, data), 'freq')

    def __sub__(self, data):
        return subtract((self, data), 'freq')

    def __mul__(self, data):
        return multiply((self, data), 'freq')

    def __truediv__(self, data):
        return divide((self, data), 'freq')

    def __pow__(self, data):
        return power((self, data), 'freq')


class Signal(FrequencyData, TimeData):
    """Class for audio signals.

    Objects of this class contain data which is directly convertable between
    time and frequency domain (equally spaced samples and frequency bins).

    """
    def __init__(
            self,
            data,
            sampling_rate,
            n_samples=None,
            domain='time',
            fft_norm=None,
            comment=None,
            dtype=np.double):
        """Create Signal with data, and sampling rate.

        Parameters
        ----------
        data : ndarray, double
            Raw data of the signal in the time or frequency domain. The memory
            layout of data is 'C'. E.g. data of ``shape = (3, 2, 1024)`` has
            3 x 2 channels with 1024 samples or frequency bins each. Frequency
            data must be provided as single sided spectra, i.e., for all
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
        dtype : string, optional
            Raw data type of the audio object. The default is `float64`

        References
        ----------
        .. [#] J. Ahrens, C. Andersson, P. Höstmad, and W. Kropp, “Tutorial on
               Scaling of the Discrete Fourier Transform and the Implied
               Physical Units of the Spectra of Time-Discrete Signals,” Vienna,
               Austria, May 2020, p. e-Brief 600.

        """

        # initialize global parameter and valid parameter spaces
        _Audio.__init__(self, domain, comment, dtype)

        # initialize signal specific parameters
        self._sampling_rate = sampling_rate

        # initialize domain specific parameters
        if domain == 'time':
            self._data = np.atleast_2d(np.asarray(data, dtype=dtype))
            self._n_samples = self._data.shape[-1]

            if fft_norm is None:
                fft_norm = 'none'
            if fft_norm in self._VALID_FFT_NORMS:
                self._fft_norm = fft_norm
            else:
                raise ValueError(("Invalid FFT normalization. Has to be "
                                  f"{', '.join(self._VALID_FFT_NORMS)}, but "
                                  f"found '{fft_norm}'"))

            TimeData.__init__(self, data, self.times, comment, dtype)
        elif domain == 'freq':
            self._data = np.atleast_2d(np.asarray(data, dtype=complex))

            n_bins = self._data.shape[-1]
            if n_samples is None:
                warnings.warn(
                    "Number of time samples not given, assuming an even "
                    "number of samples from the number of frequency bins.")
                n_samples = (n_bins - 1)*2
            elif n_samples > 2 * n_bins - 1:
                raise ValueError(("n_samples can not be larger than "
                                  "2 * data.shape[-1] - 2"))
            self._n_samples = n_samples

            FrequencyData.__init__(self, data, self.frequencies, fft_norm,
                                   comment, dtype)
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
        """Return the data in the frequency domain."""
        # converts the data from 'time' to 'freq'
        self.domain = 'freq'

        return super().freq

    @freq.setter
    def freq(self, value):
        """Set the data in the frequency domain."""
        # The intuitive version does not work: super().freq = value
        super(Signal, type(self)).freq.fset(self, value)

        # set n_samples in case of a Signal object
        if self._data.shape[-1] == self.n_bins:
            n_samples = self.n_samples
        else:
            warnings.warn(UserWarning((
                "Number of frequency bins changed, assuming an even "
                "number of samples from the number of frequency bins.")))
            n_samples = (self._data.shape[-1] - 1)*2
        self._n_samples = n_samples

    @_Audio.domain.setter
    def domain(self, new_domain):
        """Set the domain of the signal."""
        if new_domain not in self._VALID_DOMAINS:
            raise ValueError("Incorrect domain, needs to be time/freq.")

        if not (self._domain == new_domain):
            # Only process if we change domain
            if new_domain == 'time':
                # If the new domain should be time, we had a saved spectrum
                # and need to do an inverse Fourier Transform
                self.time = fft.irfft(
                    self._data, self.n_samples, self._sampling_rate,
                    self._fft_norm)
            elif new_domain == 'freq':
                # If the new domain should be freq, we had sampled time data
                # and need to do a Fourier Transform
                self.freq = fft.rfft(
                    self._data, self.n_samples, self._sampling_rate,
                    self._fft_norm)

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

    @FrequencyData.fft_norm.setter
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

        # apply new normalization if Signal is in frequency domain
        if self._fft_norm != value and self._domain == 'freq':
            # de-normalize
            self._data = fft.normalization(
                self._data, self._n_samples, self._sampling_rate,
                self._fft_norm, inverse=True)
            # normalize
            self._data = fft.normalization(
                self._data, self._n_samples, self._sampling_rate,
                value, inverse=False)

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
                      fft_norm=self.fft_norm, comment=self.comment,
                      dtype=self.dtype)
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
        if self.fft_norm == 'none':
            stype = 'energy'
        elif self.fft_norm in [
                "unitary", "amplitude", "rms", "power", "psd"]:
            stype = 'power'
        else:
            raise ValueError("No valid fft norm set.")
        return stype

    def __repr__(self):
        """String representation of signal class.
        """
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
            fft_norm=signal.fft_norm,
            dtype=signal.dtype)

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
        together.
    domain : ``'time'``, ``'freq'``, optional
        Flag to indicate if the operation should be performed in the time or
        frequency domain. If working in the frequency domain, the FFT
        normalization is removed before the operation (See
        :py:func:`pyfar.dsp.fft.normalization`). The default is ``'freq'``.

    Returns
    -------
    results : Signal, TimeData, FrequencyData, numpy array
        Result of the operation as numpy array, if `data` contains only array
        likes and numbers. Result as pyfar audio object if `data` contains an
        audio object. The `fft_norm` is ``'none'`` if all FFT norms are
        ``'none'``. Otherwise the first `fft_norm` that is not ``'none'`` is
        used.

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
        together.
    domain : ``'time'``, ``'freq'``, optional
        Flag to indicate if the operation should be performed in the time or
        frequency domain. If working in the frequency domain, the FFT
        normalization is removed before the operation (See
        :py:func:`~pyfar.dsp.fft.normalization`). The default is ``'freq'``.

    Returns
    -------
    results : Signal, TimeData, FrequencyData, numpy array
        Result of the operation as numpy array, if `data` contains only array
        likes and numbers. Result as pyfar audio object if `data` contains an
        audio object. The `fft_norm` is ``'none'`` if all FFT norms are
        ``'none'``. Otherwise the first `fft_norm` that is not ``'none'`` is
        used.
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
        together.
    domain : ``'time'``, ``'freq'``, optional
        Flag to indicate if the operation should be performed in the time or
        frequency domain. If working in the frequency domain, the FFT
        normalization is removed before the operation (See
        :py:func:`~pyfar.dsp.fft.normalization`). The default is ``'freq'``.

    Returns
    -------
    results : Signal, TimeData, FrequencyData, numpy array
        Result of the operation as numpy array, if `data` contains only array
        likes and numbers. Result as pyfar audio object if `data` contains an
        audio object. The `fft_norm` is ``'none'`` if all FFT norms are
        ``'none'``. Otherwise the first `fft_norm` that is not ``'none'`` is
        used.
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
        together.
    domain : ``'time'``, ``'freq'``, optional
        Flag to indicate if the operation should be performed in the time or
        frequency domain. If working in the frequency domain, the FFT
        normalization is removed before the operation (See
        :py:func:`~pyfar.dsp.fft.normalization`). The default is ``'freq'``.

    Returns
    -------
    results : Signal, TimeData, FrequencyData, numpy array
        Result of the operation as numpy array, if `data` contains only array
        likes and numbers. Result as pyfar audio object if `data` contains an
        audio object. The `fft_norm` is ``'none'`` if all FFT norms are
        ``'none'``. Otherwise the first `fft_norm` that is not ``'none'`` is
        used.
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
        do not work together.
    domain : ``'time'``, ``'freq'``, optional
        Flag to indicate if the operation should be performed in the time or
        frequency domain. If working in the frequency domain, the FFT
        normalization is removed before the operation (See
        :py:func:`~pyfar.dsp.fft.normalization`). The default is ``'freq'``.

    Returns
    -------
    results : Signal, TimeData, FrequencyData, numpy array
        Result of the operation as numpy array, if `data` contains only array
        likes and numbers. Result as pyfar audio object if `data` contains an
        audio object. The `fft_norm` is ``'none'`` if all FFT norms are
        ``'none'``. Otherwise the first `fft_norm` that is not ``'none'`` is
        used.
    """
    return _arithmetic(data, domain, _power)


def _arithmetic(data: tuple, domain: str, operation: Callable):
    """Apply arithmetic operations."""

    # check input and obtain meta data of new signal
    sampling_rate, n_samples, fft_norm, times, frequencies, audio_type = \
        _assert_match_for_arithmetic(data, domain)

    # apply arithmetic operation
    result = _get_arithmetic_data(data[0], n_samples, domain)

    for d in range(1, len(data)):
        result = operation(
            result, _get_arithmetic_data(data[d], n_samples, domain))

    # check if to return an audio object
    if audio_type == Signal:
        # apply desired fft normalization
        if domain == 'freq':
            result = fft.normalization(result, n_samples, sampling_rate,
                                       fft_norm)

        result = Signal(
            result, sampling_rate, n_samples, domain, fft_norm=fft_norm)
    elif audio_type == TimeData:
        result = TimeData(result, times)
    elif audio_type == FrequencyData:
        result = FrequencyData(result, frequencies, fft_norm)

    return result


def _assert_match_for_arithmetic(data: tuple, domain: str):
    """Check if type and meta data of input is fine for arithmetic operations.

    Check if sampling rate and number of samples agree if multiple signals are
    provided. Check if arrays are numeric. Check if a power signal is contained
    in the input.

    Input:
    data : tuple
        Can contain Signal, TimeData, FrequencyData, and array like data
    domain : str
        Domain in which the arithmetic operation should be performed. 'time' or
        'freq'.

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
    fft_norm = None
    times = None
    frequencies = None
    audio_type = type(None)

    # check input types and meta data
    found_audio_data = False
    for d in data:
        if isinstance(d, (Signal, TimeData, FrequencyData)):
            # store meta data upon first appearance
            if not found_audio_data:
                if isinstance(d, Signal):
                    sampling_rate = d.sampling_rate
                    n_samples = d.n_samples
                    fft_norm = d.fft_norm
                elif isinstance(d, TimeData):
                    if domain != "time":
                        raise ValueError("The domain must be 'time'.")
                    times = d.times
                elif isinstance(d, FrequencyData):
                    if domain != "freq":
                        raise ValueError("The domain must be 'freq'.")
                    frequencies = d.frequencies
                    fft_norm = d.fft_norm

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
                    # if there is a power signal, the returned signal will be
                    # a power signal
                    if d.fft_norm != 'none' and fft_norm == 'none':
                        fft_norm = d.fft_norm
                elif isinstance(d, TimeData):
                    if not np.allclose(times, d.times, atol=1e-15):
                        raise ValueError(
                            "The times does not match.")
                elif isinstance(d, FrequencyData):
                    if not np.allclose(
                            frequencies, d.frequencies, atol=1e-15):
                        raise ValueError(
                            "The frequencies do not match.")
                    if fft_norm != d.fft_norm:
                        raise ValueError(
                            "The FFT norm does not match.")

        # check type of non signal input
        else:
            dtypes = ['int8', 'int16', 'int32', 'int64',
                      'float32', 'float64',
                      'complex64', 'complex128']
            if np.asarray(d).dtype not in dtypes:
                raise ValueError(
                    f"Input must be of type Signal, {', '.join(dtypes)}")
            if np.asarray(d).dtype in ['complex64', 'complex128'] \
                    and domain == 'time':
                raise ValueError(
                    "Complex input can not be applied in the time domain.")

    return sampling_rate, n_samples, fft_norm, times, frequencies, audio_type


def _get_arithmetic_data(data, n_samples, domain):
    """
    Return data in desired domain without any fft normalization.

    Parameters
    ----------
    data : Signal, array like, number
        Input data
    n_samples :
        Number of samples of data if data is a Signal (required for fft
        normalization).
    domain : 'time', 'freq'
        Domain in which the data is returned

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
            data_out = data.freq.copy()

            if isinstance(data, Signal):
                if data.fft_norm != 'none':
                    # remove current fft normalization
                    data_out = fft.normalization(
                        data_out, n_samples, data.sampling_rate,
                        data.fft_norm, inverse=True)

        else:
            raise ValueError(
                f"domain must be 'time' or 'freq' but found {domain}")

    else:
        data_out = np.asarray(data)

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

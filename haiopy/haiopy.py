import warnings
import numpy as np
from haiopy import fft as fft


class Audio(object):
    """Abstract class for audio objects."""

    def __init__(self):
        """TODO: to be defined1. """


class Signal(Audio):
    """Class for audio signals.
    Objects of this class contain data which is directly convertable between
    time and frequency domain. Equally spaced samples or frequency bins,
    respectively.

    Attributes
    ----------
    data : ndarray, double
        Raw data of the signal
    sampling_rate : double
        Sampling rate in Hertz
    domain : string
        Domain of data ('freq'/'time')
    dtype : string
        Raw data type of the signal, optional

    """
    def __init__(
            self,
            data,
            sampling_rate,
            n_samples=None,
            domain='time',
            signal_type='energy',
            dtype=np.double):
        """Init Signal with data, sampling rate and domain and signal type.

        Attributes
        ----------
        data : ndarray, double
            Raw data of the signal
        sampling_rate : double
            Sampling rate in Hertz
        domain : string
            Domain of data ('freq'/'time')
        signal_type : string
            Distinguish between power and energy signals
        dtype : string
            Raw data type of the signal, optional
        position : Coordinates
            Coordinates object
        orientations : Orientations
            Orientations object
        """

        Audio.__init__(self)
        self._sampling_rate = sampling_rate
        self._dtype = dtype

        self._VALID_SIGNAL_TYPE = ["power", "energy"]
        if (signal_type in self._VALID_SIGNAL_TYPE) is True:
            self._signal_type = signal_type
        else:
            raise ValueError("Not a valid signal type ('power'/'energy')")

        self._VALID_SIGNAL_DOMAIN = ["time", "freq"]
        if domain in self._VALID_SIGNAL_DOMAIN:
            self._domain = domain
        else:
            raise ValueError("Invalid domain. Has to be 'time' or 'freq'.")

        if domain == 'time':
            self._data = np.atleast_2d(np.asarray(data, dtype=dtype))
            self._n_samples = self._data.shape[-1]
        elif domain == 'freq':
            if n_samples is None:
                warnings.warn(
                    "Number of time samples not given, assuming an even "
                    "number of samples from the number of frequency bins.")
                n_bins = data.shape[-1]
                n_samples = (n_bins - 1)*2
            self._n_samples = n_samples
            self._data = np.atleast_2d(np.asarray(data, dtype=np.complex))

    @property
    def domain(self):
        """The domain the data is stored in"""
        return self._domain

    @domain.setter
    def domain(self, new_domain):
        if new_domain not in self._VALID_SIGNAL_DOMAIN:
            raise ValueError("Incorrect domain, needs to be time/freq.")

        if not (self._domain == new_domain):
            # Only process if we change domain
            if new_domain == 'time':
                # If the new domain should be time, we had a saved spectrum
                # and need to do an inverse Fourier Transform
                self.time = fft.irfft(
                    self._data, self.n_samples, signal_type=self.signal_type)
            elif new_domain == 'freq':
                # If the new domain should be freq, we had sampled time data
                # and need to do a Fourier Transform
                self.freq = fft.rfft(
                    self._data, self.n_samples, signal_type=self.signal_type)

    @property
    def n_samples(self):
        """Number of samples."""
        return self._n_samples

    @property
    def n_bins(self):
        """Number of frequency bins."""
        return fft._n_bins(self.n_samples)

    @property
    def frequencies(self):
        """Frequencies of the discrete signal spectrum."""
        return np.atleast_1d(fft.rfftfreq(self.n_samples, self.sampling_rate))

    @property
    def times(self):
        """Time instances the signal is sampled at."""
        return np.atleast_1d(np.arange(0, self.n_samples) / self.sampling_rate)

    @property
    def time(self):
        """The signal data in the time domain."""
        self.domain = 'time'
        return self._data

    @time.setter
    def time(self, value):
        data = np.atleast_2d(value)
        self._domain = 'time'
        self._data = data
        self._n_samples = data.shape[-1]

    @property
    def freq(self):
        """The signal data in the frequency domain."""
        self.domain = 'freq'
        return self._data

    @freq.setter
    def freq(self, value):
        spec = np.atleast_2d(value)
        new_num_bins = spec.shape[-1]
        if new_num_bins == self.n_bins:
            n_samples = self.n_samples
        else:
            warnings.warn("Number of frequency bins different will change, "
                          "assuming an even number of samples from the number "
                          "of frequency bins.")
            n_samples = (new_num_bins - 1)*2

        self._data = spec
        self._n_samples = n_samples
        self._domain = 'freq'

    @property
    def sampling_rate(self):
        """The sampling rate of the signal."""
        return self._sampling_rate

    @sampling_rate.setter
    def sampling_rate(self, value):
        self._sampling_rate = value

    @property
    def signal_type(self):
        """The signal type."""
        return self._signal_type

    @signal_type.setter
    def signal_type(self, value):
        if (value in self._VALID_SIGNAL_TYPE) is True:
            self._signal_type = value
        else:
            raise ValueError("Not a valid signal type ('power'/'energy')")

    @property
    def dtype(self):
        """The data type of the signal. This can be any data type and precision
        supported by numpy."""
        return self._dtype

    @property
    def signal_length(self):
        """The length of the signal in seconds."""
        return (self.n_samples - 1) / self.sampling_rate

    @property
    def shape(self):
        """Shape of the data."""
        return self._data.shape[:-1]

    def __repr__(self):
        """String representation of signal class.
        """
        repr_string = (
            "Audio Signal\n"
            "--------------------\n"
            "{} channels with {} samples @ {} Hz sampling rate".format(
                self.shape, self.n_samples, self._sampling_rate))
        return repr_string

    def __getitem__(self, key):
        """Get signal channels at key.
        """
        if isinstance(key, (int, slice, tuple)):
            try:
                data = self._data[key]
            except KeyError:
                raise KeyError("Index is out of bounds")
        else:
            raise TypeError(
                    "Index must be int, not {}".format(type(key).__name__))
        items = Signal(
            data,
            sampling_rate=self.sampling_rate,
            domain=self.domain,
            signal_type=self.signal_type,
            dtype=self.dtype)

        return items

    def __setitem__(self, key, value):
        """Set signal channels at key.
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

    def __len__(self):
        """Length of the object which is the number of samples stored.
        """
        return self.n_samples

    def _assert_matching_meta_data(self, other):
        """Check if the sampling rate, the number of samples, and the signal
        type of two Signal objects match.
        """
        if not isinstance(other, Signal):
            raise ValueError("Comparison only valid against Signal objects.")
        if self.sampling_rate != other.sampling_rate:
            raise ValueError("The sampling rates do not match.")
        if self.n_samples != other.n_samples:
            raise ValueError("The number of samples does not match.")
        if self.signal_type != other.signal_type:
            raise ValueError("The signal types do not match.")

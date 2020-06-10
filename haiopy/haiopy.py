import warnings

import numpy as np

from haiopy import fft as fft
from haiopy.coordinates import Coordinates
from haiopy.orientation import Orientation


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
    position : Coordinates
        Coordinates object
    orientation : Orientation
        Orientation object


    """
    def __init__(self,
                 data,
                 sampling_rate,
                 n_samples=None,
                 domain='time',
                 signal_type='energy',
                 dtype=np.double,
                 position=Coordinates(),
                 orientation=Orientation()):
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
        orientation : Orientation
            Orientation object
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
                warnings.warn("Number of time samples not given, assuming an\
                    even number of samples from the number of frequency bins.")
                n_bins = data.shape[-1]
                n_samples = (n_bins - 1)*2
            self._n_samples = n_samples
            self._data = np.atleast_2d(np.asarray(data, dtype=np.complex))

        if isinstance(position, Coordinates):
            self._position = position
        else:
            raise TypeError(("Input value has to be coordinates object, "
                             "not {}").format(type(position).__name__))

        if isinstance(orientation, Orientation):
            self._orientation = orientation
        else:
            raise TypeError(("Input value has to be orientation object, "
                             "not {}").format(type(orientation).__name__))

    @property
    def n_samples(self):
        """Number of samples."""
        return self._data.shape[-1]

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
        return self._data

    @time.setter
    def time(self, value):
        self._data = np.atleast_2d(value)

    @property
    def freq(self):
        """The signal data in the frequency domain."""
        freq = fft.rfft(self._data, self.n_samples, self._signal_type)
        return freq

    @freq.setter
    def freq(self, value):
        self._data = np.atleast_2d(
            fft.irfft(value, self.n_samples, self.signal_type))

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
    def position(self):
        """Coordinates of the object."""
        return self._position

    @position.setter
    def position(self, value):
        if isinstance(value, Coordinates):
            self._position = value
        else:
            raise TypeError(("Input value has to be coordinates object, "
                             "not {}").format(type(value).__name__))

    @property
    def orientation(self):
        """Orientation of the object."""
        return self._orientation

    @orientation.setter
    def orientation(self, value):
        if isinstance(value, Orientation):
            self._orientation = value
        else:
            raise TypeError(("Input value has to be orientation object, "
                             "not {}").format(type(value).__name__))

    @property
    def shape(self):
        """Shape of the data."""
        return self._data.shape

    def __repr__(self):
        """String representation of signal class.
        """
        repr_string = ("Audio Signal\n"
                       "--------------------\n"
                       "Dimensions: {}x{}\n"
                       "Sampling rate: {} Hz\n"
                       "Signal type: {}\n"
                       "Signal length: {} sec").format(
                       self.shape[0], self.n_samples, self._sampling_rate,
                       self._signal_type, self.signal_length)
        return repr_string

    def __getitem__(self, key):
        """Get signal channels at key.
        """
        if isinstance(key, (int, slice)):
            try:
                return self._data[key]
            except KeyError:
                raise KeyError("Index is out of bounds")
        elif isinstance(key, tuple):
            try:
                return self._data[key]
            except KeyError:
                raise KeyError("Index is out of bounds")
        else:
            raise TypeError(
                    "Index must be int, not {}".format(type(key).__name__))

    def __setitem__(self, key, value):
        """Set signal channels at key.
        """
        if isinstance(key, (int, slice)):
            try:
                self._data[key] = value
            except KeyError:
                raise KeyError("Index is out of bound")
        else:
            raise TypeError(
                    "Index must be int, not {}".format(type(key).__name__))

    def __len__(self):
        """Length of the object which is the number of samples stored.
        """
        return self.n_samples

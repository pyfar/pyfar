import numpy as np

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
    samplingrate : double
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
                 samplingrate,
                 domain='time',
                 signaltype='power',
                 dtype=None,
                 position=Coordinates(),
                 orientation=Orientation()):
        """Init Signal with data, sampling rate and domain and signal type.

        Attributes
        ----------
        data : ndarray, double
            Raw data of the signal
        samplingrate : double
            Sampling rate in Hertz
        domain : string
            Domain of data ('freq'/'time')
        signaltype : string
            Destinguish between power and energy signals
        dtype : string
            Raw data type of the signal, optional
        position : Coordinates
            Coordinates object
        orientation : Orientation
            Orientation object
        """

        Audio.__init__(self)
        self._samplingrate = samplingrate
        if domain == 'time':
            if not dtype:
                self._dtype = data.dtype
            else:
                self._dtype = dtype
            self._data = np.asarray(data, dtype=dtype)
        elif domain == 'freq':
            if dtype is None:
                self._dtype = np.double
            if self.iscomplex:
                self._data = np.asarray(np.fft.ifft(data), dtype=dtype)
            else:
                self._data = np.asarray(np.fft.irfft(data), dtype=dtype)

        self._VALID_SIGNALTYPE = ["power", "energy"]
        if (signaltype in self._VALID_SIGNALTYPE) is True:
            self._signaltype = signaltype
        else:
            raise ValueError("Not a valid signal type ('power'/'energy')")

        if type(position).__name__ == "Coordinates":
            self._position = position
        else:
            raise TypeError(("Input value has to be coordinates object, "
                             "not {}").format(type(position).__name__))

        if type(orientation).__name__ == "Orientation":
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
        return (self._data.shape[-1] / 2) + 1

    @property
    def frequencies(self):
        """Frequencies of the discrete signal spectrum."""
        return np.fft.rfftfreq(self.n_samples, d=1/self.samplingrate)

    @property
    def times(self):
        """Time instances the signal is sampled at."""
        return np.arange(0, self.n_samples) / self.samplingrate

    @property
    def time(self):
        """The signal data in the time domain."""
        return self._data

    @time.setter
    def time(self, value):
        self._data = value

    @property
    def freq(self):
        """The signal data in the frequency domain."""
        if self.iscomplex:
            freq = np.fft.rfft(self._data)
        else:
            freq = np.fft.fft(self._data)
        return freq

    @freq.setter
    def freq(self, value):
        if self.iscomplex:
            self.data = np.fft.ifft(value)
        else:
            self.data = np.fft.irfft(value)

    @property
    def samplingrate(self):
        """The sampling rate of the signal."""
        return self._samplingrate

    @samplingrate.setter
    def samplingrate(self, value):
        self._samplingrate = value

    @property
    def signaltype(self):
        """The signal type."""
        return self._signaltype

    @signaltype.setter
    def signaltype(self, value):
        if (value in self._VALID_SIGNALTYPE) is True:
            self._signaltype = value
        else:
            raise ValueError("Not a valid signal type ('power'/'energy')")

    @property
    def dtype(self):
        """The data type of the signal. This can be any data type and precision
        supported by numpy."""
        return self._dtype

    @property
    def signallength(self):
        """The length of the signal in seconds."""
        return (self.n_samples - 1) / self.samplingrate

    @property
    def position(self):
        """Coordinates of the object."""
        return self._position

    @position.setter
    def position(self, value):
        if type(value).__name__ == "Coordinates":
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
        self._orientation = value

    @property
    def shape(self):
        """Shape of the data."""
        return self._data.shape

    @property
    def iscomplex(self):
        if 'complex' in str(self.dtype):
            iscomplex = True
        else:
            iscomplex = False
        return iscomplex

    def __repr__(self):
        """String representation of signal class.
        """
        if len(self.shape) == 1:
            n_channels = 1
        else:
            n_channels = self.shape[0]

        repr_string = ("Audio Signal\n"
                       "--------------------\n"
                       "Dimensions: {}x{}\n"
                       "Sampling rate: {} Hz\n"
                       "Signal type: {}\n"
                       "Signal length: {} sec").format(
                       n_channels, self.n_samples, self._samplingrate,
                       self._signaltype, self.signallength)
        return repr_string

    def __getitem__(self, key):
        """Get signal channels at key.
        """
        if isinstance(key, (int, slice)):
            try:
                return self._data[key]
            except KeyError:
                raise KeyError("Index is out of range")
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
                raise KeyError("Index is out of range")
        else:
            raise TypeError(
                    "Index must be int, not {}".format(type(key).__name__))

    def __len__(self):
        """Length of the object which is the number of samples stored.
        """
        return self.n_samples
"""
The following documents the audio classes and arithmetic operations for
audio data. More details and background is given in the gallery
(:doc:`audio objects<gallery:gallery/interactive/pyfar_audio_objects>`,
:doc:`Fourier transform<gallery:gallery/interactive/fast_fourier_transform>`,
:doc:`gallery:gallery/interactive/pyfar_arithmetics`).

All Audio objects support indexing, for example you can get a copy of the first
channel of an audio object with

>>> import pyfar as pf
>>> signal = pf.signals.noise(10, rms=[1, 1])
>>> first_channel = signal[0]

and set the first channel with

>>> signal[0] = pf.signals.noise(10, rms=2)

For more information see the `NumPy documentation on indexing
<https://numpy.org/doc/stable/user/basics.indexing.html>`_.

In addition `Signal` objects support iteration across the first dimension.
The actual iteration is handled through numpy's array iteration. The following
iterates the channels of a `Signal` object

>>> signal = pf.signals.impulse(2, amplitude=[1, 1, 1])
>>> for idx, channel in enumerate(signal):
>>>     channel.time *= idx
>>>     signal[idx] = channel
"""

from copy import deepcopy
import warnings
import deepdiff
import numpy as np
import pyfar.dsp.fft as fft
from typing import Callable
from pyfar.classes.warnings import PyfarDeprecationWarning


class _Audio():
    """
    Abstract class for audio objects.

    This class holds all the methods and properties that are common to its
    three sub-classes :py:func:`TimeData`, :py:func:`FrequencyData`, and
    :py:func:`Signal`.
    """

    # indicate use of _Audio arithmetic operations for
    # overloaded operators (e.g. __rmul__)
    __array_priority__ = 1.0

    def __init__(self, domain, comment=""):

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

    @property
    def cdim(self):
        """
        Return channel dimension.

        The channel dimension (`cdim`) gives the number of dimensions of the
        audio data excluding the last dimension, which is `n_samples` for
        time domain objects and `n_bins` for frequency domain objects.
        Therefore it is equivalent to the length of the channel shape
        (`cshape`) (e.g. ``self.cshape = (2, 3)``; ``self.cdim = 2``).
        """
        return len(self.cshape)

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
        except ValueError as e:
            if np.prod(newshape) != np.prod(self.cshape):
                raise ValueError(
                    (f"Cannot reshape audio object of cshape "
                     f"{self.cshape} to {newshape}")) from e

        return reshaped

    def transpose(self, *axes):
        """Transpose time/frequency data and return copy of the audio object.

        Parameters
        ----------
        axes : empty, ``None``, iterable of ints, or n ints
            Define how the :py:mod:` caxes <pyfar._concepts.audio_classes>`
            are ordered in the transposed audio object.
            Note that the last dimension of the data in the audio object
            always contains the time samples or frequency bins and can not
            be transposed.

            empty (default) or ``None``
                reverses the order of ``self.caxes``.
            iterable of ints
                `i` in the `j`-th place of the interable means
                that the `i`-th caxis becomes transposed object's `j`-th caxis.
            n ints
                same as 'iterable of ints'.
        """
        if hasattr(axes, '__iter__'):
            axes = axes[0] if len(axes) == 1 else axes
        if axes is None or len(axes) == 0:
            axes = tuple(range(len(self.cshape)))[::-1]
        else:
            assert all(a > -len(self.cshape) - 1 for a in axes), \
                "Negative axes index out of bounds."
            axes = tuple([a % len(self.cshape) if a < 0 else a for a in axes])

        # throw exception before deepcopy
        np.empty(np.ones(len(self.cshape), dtype=int)).transpose(axes)
        transposed = deepcopy(self)
        transposed._data = transposed._data.transpose(*axes, len(self.cshape))

        return transposed

    @property
    def T(self):
        """Shorthand for `Signal.transpose()`."""
        return self.transpose()

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
        if not isinstance(value, str):
            raise TypeError("comment has to be of type string.")
        else:
            self._comment = value

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

        # add empty slice at the end to always get all data contained in last
        # dimension (samples or frequency bins)
        if hasattr(key, '__iter__'):
            key = (*key, slice(None))

        # try indexing and raise verbose errors if it fails
        try:
            data = self._data[key]
        except IndexError as e:
            if 'too many indices for array' in str(e):
                raise IndexError((
                    f'Indexed dimensions must not exceed the channel '
                    f'dimension (cdim), which is {len(self.cshape)}')) from e
            else:
                raise e

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

    @staticmethod
    def _check_input_type_is_numeric(data: np.ndarray):
        """
        Check if input data is numeric and raise TypeError if not.
        """
        # check if data type is numeric
        if data.dtype.kind not in ["u", "i", "f", "c"]:
            raise TypeError((f"The input data is {data.dtype} must be int, "
                             "uint, float, or complex"))

    @staticmethod
    def _check_input_values_are_numeric(data: np.ndarray):
        """
        Check if input data contains only numeric values and raise TypeError
        if not. This is only required for Signal objects but is kept here
        to improve readability.
        """
        # check for non-numeric values
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            raise ValueError((
                "The input values must be numeric but contain at "
                "least one non-numerical value (inf or NaN)"))


class TimeData(_Audio):
    """
    Create audio object with time data and times.

    Objects of this class contain time data which is not directly convertible
    to frequency domain, i.e., non-equidistant samples.

    Parameters
    ----------
    data : array, double
        Raw data in the time domain. The memory layout of data is 'C'. E.g.,
        data of ``shape = (3, 2, 1024)`` has 3 x 2 channels with 1024 samples
        each. The data can be ``int`` or ``float`` and is converted to
        ``float`` in any case.
    times : array, double
        Times in seconds at which the data is sampled. The number of times
        must match the `size` of the last dimension of `data`.
    comment : str, optional
        A comment related to `data`. The default is ``""``, which initializes
        an empty string.
    is_complex : bool, optional
        A flag which indicates if the time data are real or complex-valued.
        The default is ``False``.
    """

    def __init__(self, data, times, comment="", is_complex=False):
        """Create TimeData object with data, and times."""

        _Audio.__init__(self, 'time', comment)

        if not isinstance(is_complex, bool):
            raise TypeError("``is_complex`` flag is "
                            f"{type(is_complex).__name__}"
                            "but must be a boolean")

        self._complex = is_complex
        self.time = data

        self._times = np.atleast_1d(np.asarray(times).flatten())
        if self._times.size != self.n_samples:
            raise ValueError(
                "The length of times must be data.shape[-1]")
        if np.any(np.diff(self._times) <= 0) and len(self._times) > 1:
            raise ValueError("Times must be monotonously increasing.")

    @property
    def time(self):
        """Return or set the data in the time domain."""
        return self._data

    @time.setter
    def time(self, value):
        """Return or set the time data."""
        # check and set the data and meta data
        data = np.atleast_2d(np.asarray(value))
        self._check_input_type_is_numeric(data)
        if self.complex:
            data = np.atleast_2d(np.asarray(value, dtype=complex))
        else:
            if data.dtype.kind in ["i", "u"]:
                data = np.atleast_2d(np.asarray(value, dtype=float))
            elif data.dtype.kind == "c":
                raise ValueError("time data is complex, set is_complex "
                                 "flag or pass real-valued data.")

        self._data = data
        self._n_samples = data.shape[-1]
        # setting the domain is only required for Signal. Setting it here
        # avoids the need for overloading the setter and does not harm TimeData
        self._domain = 'time'

    @property
    def complex(self):
        """Return or set the flag indicating if the time data is complex."""
        return self._complex

    @complex.setter
    def complex(self, value):
        # set from complex=True to complex=False
        if self._complex and not value:
            if np.all(np.abs(np.imag(self._data))) >= 1e-14:
                raise ValueError("Signal has complex-valued time data"
                                 " is_complex flag cannot be `False`.")
            self._complex = value
            self._data = self._data.astype(float)
        # from complex=False to complex=True
        if not self._complex and value:
            self._complex = value
            self._data = self._data.astype(complex)

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
            data, times=self.times, comment=self.comment,
            is_complex=self.complex)
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
        """Add two TimeData objects."""
        return add((self, data), 'time')

    def __radd__(self, data):
        """Add two TimeData objects."""
        return add((data, self), 'time')

    def __sub__(self, data):
        """Subtract two TimeData objects."""
        return subtract((self, data), 'time')

    def __rsub__(self, data):
        """Subtract two TimeData objects."""
        return subtract((data, self), 'time')

    def __mul__(self, data):
        """Multiply two TimeData objects."""
        return multiply((self, data), 'time')

    def __rmul__(self, data):
        """Multiply two TimeData objects."""
        return multiply((data, self), 'time')

    def __truediv__(self, data):
        """Divide two TimeData objects."""
        return divide((self, data), 'time')

    def __rtruediv__(self, data):
        """Divide two TimeData objects."""
        return divide((data, self), 'time')

    def __pow__(self, data):
        """Raise two TimeData objects to the power."""
        return power((self, data), 'time')

    def __rpow__(self, data):
        """Raise two TimeData objects to the power."""
        return power((data, self), 'time')

    def __matmul__(self, data):
        """Matrix multiplication of two TimeData objects."""
        return matrix_multiplication(
            (self, data), 'time')

    def __rmatmul__(self, data):
        """Matrix multiplication of two TimeData objects."""
        return matrix_multiplication(
            (data, self), 'time')


class FrequencyData(_Audio):
    """
    Create audio object with frequency data and frequencies.

    Objects of this class contain frequency data which is not directly
    convertible to the time domain, i.e., non-equidistantly spaced bins or
    incomplete spectra.

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
        A comment related to the data. The default is ``""``, which
        initializes an empty string.


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

    def __init__(self, data, frequencies, comment=""):
        """Create audio object with frequency data and frequencies."""

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
        """Return or set the data in the frequency domain."""
        return self._data

    @freq.setter
    def freq(self, value):
        """Return or set the data in the frequency domain."""

        # check data type
        data = np.atleast_2d(np.asarray(value))
        self._check_input_type_is_numeric(data)
        if data.dtype.kind in ["i", "u"]:
            data = data.astype("float")

        # match shape of frequencies
        if self.frequencies.size != data.shape[-1]:
            raise ValueError(
                "Number of frequency values does not match the number of "
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
        """Add two FrequencyData objects."""
        return add((self, data), 'freq')

    def __radd__(self, data):
        """Add two FrequencyData objects."""
        return add((data, self), 'freq')

    def __sub__(self, data):
        """Subtract two FrequencyData objects."""
        return subtract((self, data), 'freq')

    def __rsub__(self, data):
        """Subtract two FrequencyData objects."""
        return subtract((data, self), 'freq')

    def __mul__(self, data):
        """Multiply two FrequencyData objects."""
        return multiply((self, data), 'freq')

    def __rmul__(self, data):
        """Multiply two FrequencyData objects."""
        return multiply((data, self), 'freq')

    def __truediv__(self, data):
        """Divide two FrequencyData objects."""
        return divide((self, data), 'freq')

    def __rtruediv__(self, data):
        """Divide two FrequencyData objects."""
        return divide((data, self), 'freq')

    def __pow__(self, data):
        """Raise two FrequencyData objects to the power."""
        return power((self, data), 'freq')

    def __rpow__(self, data):
        """Raise two FrequencyData objects to the power."""
        return power((data, self), 'freq')

    def __matmul__(self, data):
        """Matrix multiplication of two FrequencyData objects."""
        return matrix_multiplication(
            (self, data), 'freq')

    def __rmatmul__(self, data):
        """Matrix multiplication of two FrequencyData objects."""
        return matrix_multiplication(
            (data, self), 'freq')


class Signal(FrequencyData, TimeData):
    """
    Create audio object with time or frequency data and sampling rate.

    Objects of this class contain data which is directly convertible between
    time and frequency domain (equally spaced samples and frequency bins). The
    data is always real valued in the time domain and complex valued in the
    frequency domain.

    Parameters
    ----------
    data : ndarray, float, complex
        Raw data of the signal in the time or frequency domain. The memory
        layout of data is 'C'. E.g. data of ``shape = (3, 2, 1024)`` has
        3 x 2 channels with 1024 samples or frequency bins each, depending
        on the specified ``domain``. Integer arrays will be converted to
        floating point precision. Note that providing complex valued time
        domain data is only possible when the parameter ``complex`` is
        ``True``. If the specified ``domain`` is ``freq`` and
        ``complex`` is ``True`` the data needs to represent a double-sided
        spectrum, otherwise the single-sided spectrum for positive
        frequencies needs to be provided.
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
    comment : str, optional
        A comment related to `data`. The default is ``""``, which
        initializes an empty string.
    is_complex : bool, optional
        Specifies if the underlying time domain data are complex
        or real-valued. If ``True`` and `domain` is ``'time'``, the
        input data will be cast to complex. The default is ``False``.

    References
    ----------
    .. [#] J. Ahrens, C. Andersson, P. Höstmad, and W. Kropp, “Tutorial on
            Scaling of the Discrete Fourier Transform and the Implied
            Physical Units of the Spectra of Time-Discrete Signals,” Vienna,
            Austria, May 2020, p. e-Brief 600.
    """

    def __init__(
            self,
            data,
            sampling_rate,
            n_samples=None,
            domain='time',
            fft_norm='none',
            comment="",
            is_complex=False):
        """
        Create audio Signal with time or frequency data and sampling rate.
        """
        # unpack array
        if hasattr(sampling_rate, '__iter__'):
            assert len(sampling_rate) != 0
            if len(sampling_rate) != 1:
                raise ValueError("Multirate signals are not supported.")
            sampling_rate = sampling_rate[0]

        # initialize signal specific parameters
        self._sampling_rate = sampling_rate

        if not isinstance(is_complex, bool):
            raise TypeError("``is_complex`` flag is "
                            f"{type(is_complex).__name__} "
                            "but must be a boolean")

        self._complex = is_complex
        self._VALID_FFT_NORMS = [
            "none", "unitary", "amplitude", "rms", "power", "psd"]

        # check fft norm
        if fft_norm in self._VALID_FFT_NORMS:
            if self._complex and fft_norm in ["rms", "power", "psd"]:
                raise ValueError(("'rms', 'power', and psd FFT normalization "
                                  "is not valid for complex time signals"))
            else:
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
            TimeData.__init__(self, data, times, comment, is_complex)
        elif domain == 'freq':
            # check and set n_samples
            if n_samples is None:
                n_samples = fft._n_samples_from_n_bins(
                    data.shape[-1], is_complex=is_complex)
                warnings.warn(
                    f"Number of samples not given, assuming {n_samples} "
                    f"samples from {data.shape[-1]} frequency bins.",
                    stacklevel=2)
            elif (n_samples > 2 * data.shape[-1] - 1) and not self.complex:
                raise ValueError(("n_samples can not be larger than "
                                  "2 * data.shape[-1] - 2"
                                  "when passing one-sided Fourier spectrum"))
            elif (n_samples > data.shape[-1]) and self.complex:
                raise ValueError(("n_samples can not be larger than "
                                  "data.shape[-1] when passing double-"
                                  "sided Fourier spectrum"))
            self._n_samples = n_samples
            # Init remaining parameters
            FrequencyData.__init__(self, data, self.frequencies, comment)
            delattr(self, '_frequencies')
        else:
            raise ValueError("Invalid domain. Has to be 'time' or 'freq'.")

        # additional input data check required for Signal objects and not done
        # in FrequencyData and TimeData __init__ methods
        self._check_input_values_are_numeric(data)

    @property
    def time(self):
        """Return or set the data in the time domain."""
        # this overrides the setter TimeData.time

        # converts the data from 'freq' to 'time'
        self.domain = 'time'

        return super().time

    @time.setter
    def time(self, value):
        """Return or set the data in the time domain."""
        # this overrides the setter TimeData.time

        # set data using parent class
        TimeData.time.fset(self, value)
        # additional check required for signal objects
        self._check_input_values_are_numeric(self.time)

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
            self.fft_norm, inverse=False,
            single_sided=not self.complex)

        return data

    @freq.setter
    def freq(self, value):
        """Return or set the normalized frequency domain data."""
        self._freq(value, raw=False)

    @property
    def freq_raw(self):
        """Return or set the frequency domain data without normalization.

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
        """Return or set the frequency domain data without normalization."""
        self._freq(value, raw=True)

    def _freq(self, value, raw):
        """Set the frequency domain data."""
        # check data type
        data = np.atleast_2d(np.asarray(value))
        self._check_input_type_is_numeric(data)
        self._check_input_values_are_numeric(data)
        # Check n_samples
        if data.shape[-1] != self.n_bins:
            self._n_samples = fft._n_samples_from_n_bins(
                data.shape[-1], self.complex)
            warnings.warn(
                f"Number of samples not given, assuming {self.n_samples} "
                f"samples from {data.shape[-1]} frequency bins.", stacklevel=2)
        # set domain
        self._domain = 'freq'
        if not raw:
            # remove normalization
            data = fft.normalization(
                data, self._n_samples, self._sampling_rate,
                self._fft_norm, inverse=True,
                single_sided=not self.complex)
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
                if self.complex:
                    # assume frequency data came from a complex-valued time
                    # signal and we have a double-sided Fourier spectrum
                    self._data = fft.ifft(
                        self._data, self.n_samples, self._sampling_rate,
                        fft_norm='none')
                else:
                    # assume frequency data came from a real-valued time signal
                    # and we have a single-sided Fourier spectrum
                    self._data = fft.irfft(
                        self._data, self.n_samples, self._sampling_rate,
                        fft_norm='none')
            elif new_domain == 'freq':
                # If the new domain should be freq, we had sampled time data
                # and need to do a Fourier Transform (without normalization)
                if self.complex:
                    # If the time data are complex-valued, calculate a
                    # double-sided Fourier spectrum
                    self._data = fft.fft(
                        self._data, self.n_samples, self._sampling_rate,
                        fft_norm='none')
                else:
                    # If the time data are real-valued, calculate a
                    # single-sided Fourier spectrum
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
    def complex(self):
        """Return or set the flag indicating if the time data is complex."""
        return self._complex

    @complex.setter
    def complex(self, value):
        # from complex=True to complex=False
        if self._complex and not value:
            if self._domain == 'time':
                # call complex setter of timeData
                super(Signal, self.__class__).complex.fset(self, value)
            if self._domain == 'freq':
                # and remove redundant part of the spectrum
                # if data are conjuagte symmetric data
                self._data = fft.remove_mirror_spectrum(self._data)
                self._complex = value
        # from complex=False to complex=True
        if not self._complex and value:
            if self._domain == 'time':
                # call complex setter of timeData
                super(Signal, self.__class__).complex.fset(self, value)
            elif self._domain == 'freq':
                # add mirror spectrum according to the "old" time data
                self._data = fft.add_mirror_spectrum(self._data,
                                                     not fft._is_odd(
                                                      self.n_samples))
                self._complex = value
        # check fft norm if complex flag was set
        if self._complex:
            if self.fft_norm in ["rms", "power", "psd"]:
                raise ValueError(("'rms', 'power', and 'psd' FFT "
                                  "normalization is not valid for complex "
                                  "time signals"))

    @property
    def times(self):
        """Time instances the signal is sampled at."""
        return np.atleast_1d(np.arange(0, self.n_samples) / self.sampling_rate)

    @property
    def frequencies(self):
        """Frequencies of the discrete signal spectrum."""
        if self.complex:
            # assume the time domain data were complex-valued
            # such that we need a two-sided Fourier spectrum
            return np.atleast_1d(fft.fftfreq(self.n_samples,
                                             self.sampling_rate))
        else:
            return np.atleast_1d(fft.rfftfreq(self.n_samples,
                                              self.sampling_rate))

    @property
    def n_bins(self):
        """Number of frequency bins."""
        return fft._n_bins_from_n_samples(self.n_samples, self.complex)

    @property
    def fft_norm(self):
        """
        The normalization for the Discrete Fourier Transform (DFT).

        See :py:func:`~pyfar.dsp.fft.normalization` and
        :ref:`arithmetic operations<gallery:/gallery/interactive/fast_fourier_transform.ipynb#FFT-normalizations>`
        for more information.
        """  # noqa: E501
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
        if self._complex and value in ["rms", "power", "psd"]:
            raise ValueError(("'rms', 'power', and 'psd' FFT normalization is "
                              "not valid for complex time signals"))
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
                      is_complex=self.complex)
        return item

    def _encode(self):
        """Return dictionary for the encoding."""
        selfcopy = self.copy()
        selfcopy.domain = "time"
        class_dict = selfcopy.__dict__
        return class_dict

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
        warnings.warn(
            ("len(Signal) will be deprecated in pyfar 0.8.0 "
             "Use Signal.n_samples instead"),
             PyfarDeprecationWarning, stacklevel=2)
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
    """Iterator for :py:func:`Signal`.
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
            is_complex=signal.complex)

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
        :ref:`arithmetic operations<gallery:/gallery/interactive/pyfar_arithmetics.ipynb#DFT-normalization-and-arithmetic-operations>`
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

    * If only one signal is involved in the operation, the result gets the same
      normalization.
    * If one signal has the FFT normalization ``'none'``, the results gets
      the normalization of the other signal.
    * If both signals have the same FFT normalization, the results gets the
      same normalization.
    * Other combinations raise an error.
    """  # noqa: E501
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
        :ref:`arithmetic operations<gallery:/gallery/interactive/pyfar_arithmetics.ipynb#DFT-normalization-and-arithmetic-operations>`
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

    * If only one signal is involved in the operation, the result gets the same
      normalization.
    * If one signal has the FFT normalization ``'none'``, the results gets
      the normalization of the other signal.
    * If both signals have the same FFT normalization, the results gets the
      same normalization.
    * Other combinations raise an error.
    """  # noqa: E501
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
        :ref:`arithmetic operations<gallery:/gallery/interactive/pyfar_arithmetics.ipynb#DFT-normalization-and-arithmetic-operations>`
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

    * If only one signal is involved in the operation, the result gets the same
      normalization.
    * If one signal has the FFT normalization ``'none'``, the results gets
      the normalization of the other signal.
    * If both signals have the same FFT normalization, the results gets the
      same normalization.
    * Other combinations raise an error.
    """  # noqa: E501
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
        :ref:`arithmetic operations<gallery:/gallery/interactive/pyfar_arithmetics.ipynb#DFT-normalization-and-arithmetic-operations>`
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

    * If only one signal is involved in the operation, the result gets the same
      normalization.
    * If the denominator signal has the FFT normalization ``'none'``, the
      result gets the normalization of the numerator signal.
    * If both signals have the same FFT normalization, the results gets the
      normalization ``'none'``.
    * Other combinations raise an error.
    """  # noqa: E501
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
        :ref:`arithmetic operations<gallery:/gallery/interactive/pyfar_arithmetics.ipynb#DFT-normalization-and-arithmetic-operations>`
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

    * If only one signal is involved in the operation, the result gets the same
      normalization.
    * If one signal has the FFT normalization ``'none'``, the results gets
      the normalization of the other signal.
    * If both signals have the same FFT normalization, the results gets the
      same normalization.
    * Other combinations raise an error.
    """  # noqa: E501
    return _arithmetic(data, domain, _power)


def matrix_multiplication(
        data: tuple, domain='freq', axes=[(-2, -1), (-2, -1), (-2, -1)]):
    """Matrix multiplication of multidimensional pyfar audio objects and/or
    array likes.

    The multiplication is based on :py:data:`numpy.matmul` and acts on the channels
    of audio objects (:py:func:`Signal`, :py:func:`TimeData`, and
    :py:func:`FrequencyData`). Alternatively, the ``@`` operator can be used
    for frequency domain matrix multiplications with the default parameters.

    Parameters
    ----------
    data : tuple of the form (data_1, data_2, ..., data_N)
        Data to be multiplied. Can contain pyfar audio objects and array likes.
        If multiple audio objects are passed they must be of the same type and
        their FFT normalizations must allow the multiplication (see
        :ref:`arithmetic operations<gallery:/gallery/interactive/pyfar_arithmetics.ipynb#DFT-normalization-and-arithmetic-operations>`
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

        In case of pyfar audio objects, the indices refer to the channel axis
        (`caxis`). It denotes an axis of the data inside an audio object but
        ignores the last axis that contains the time samples or frequency bins.
        For example, a signal with 4 times 2 channels and 120
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
    :doc:`broadcastable<numpy:user/basics.broadcasting>`
    except for the axes specified by the `axes` parameter.

    The `fft_norm` of the result is as follows

    * If only one signal is involved in the operation, the result gets the same
      normalization.
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

    """  # noqa: E501
    return _arithmetic(data, domain, _matrix_multiplication, axes=axes)


def _arithmetic(data: tuple, domain: str, operation: Callable, **kwargs):
    """Apply arithmetic operations."""
    #NOTE: The import is done here to avoid a circular import
    from pyfar.classes.transmission_matrix import TransmissionMatrix

    # check input and obtain meta data of new signal
    division = True if operation == _divide else False
    matmul = True if operation == _matrix_multiplication else False
    sampling_rate, n_samples, fft_norm, times, frequencies, audio_type, \
        cshape, contains_complex = _assert_match_for_arithmetic(
            data, domain, division, matmul)

    # apply arithmetic operation
    result = _get_arithmetic_data(
        data[0], domain, cshape, matmul, audio_type, contains_complex)

    for d in range(1, len(data)):
        if matmul:
            kwargs['audio_type'] = audio_type
        result = operation(
            result,
            _get_arithmetic_data(data[d], domain, cshape, matmul, audio_type,
                                 contains_complex),
            **kwargs)

    # check if to return an audio object
    if audio_type == Signal:
        # Set unnormalized spectrum
        result = Signal(
            result, sampling_rate, n_samples, domain, fft_norm='none',
            is_complex=contains_complex)
        # Set fft norm
        result.fft_norm = fft_norm
    elif audio_type == TimeData:
        result = TimeData(result, times, is_complex=contains_complex)
    elif audio_type == FrequencyData:
        result = FrequencyData(result, frequencies)
    elif audio_type == TransmissionMatrix:
        result = TransmissionMatrix(result, frequencies)

    return result


def _assert_match_for_arithmetic(data: tuple, domain: str, division: bool,
                                 matmul: bool):
    """Check if type and meta data of input is fine for arithmetic operations.

    Check if sampling rate and number of samples agree if multiple signals are
    provided. Check if arrays are numeric. Check if a power signal is contained
    in the input. Extract cshape of result. Check if input data is a
    complex-valued Signal or complex-valued TimeData.

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
    contains_complex: bool, False
        Indicates if input data contains a complex-valued Signal or
        complex-valued TimeData.
    """
    #NOTE: The import is done here to avoid a circular import
    from pyfar.classes.transmission_matrix import TransmissionMatrix

    # we need at least two signals
    if not isinstance(data, tuple):
        raise ValueError("Input argument 'data' must be a tuple.")

    # check validity of domain
    if domain not in ['time', 'freq']:
        raise ValueError(f"domain must be time or freq but is {domain}.")

    # properties that must match
    sampling_rate = None
    n_samples = None
    # None indicates that no audio object is yet involved in the operation
    # it will change upon detection of the first audio object
    fft_norm = None
    times = None
    frequencies = None
    audio_type = type(None)
    cshape = ()
    contains_complex = False

    # check input types and meta data
    n_audio_objects = 0
    for d in data:
        if isinstance(d, (Signal, TimeData, FrequencyData,
                          TransmissionMatrix)):
            # check for complex valued time data
            if isinstance(d, (Signal, TimeData)):
                if d.complex:
                    contains_complex = True

            # store meta data upon first appearance
            n_audio_objects += 1
            if n_audio_objects == 1:
                if isinstance(d, Signal):
                    sampling_rate = d.sampling_rate
                    n_samples = d.n_samples
                    fft_norm = d.fft_norm
                elif isinstance(d, TimeData):
                    if domain != "time":
                        raise ValueError("The domain must be 'time'.")
                    times = d.times
                elif isinstance(d, (FrequencyData, TransmissionMatrix)):
                    if domain != "freq":
                        raise ValueError("The domain must be 'freq'.")
                    frequencies = d.frequencies
                if not matmul:
                    cshape = d.cshape
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
                elif isinstance(d, (FrequencyData, TransmissionMatrix)):
                    if not np.allclose(
                            frequencies, d.frequencies, atol=1e-15):
                        raise ValueError(
                            "The frequencies do not match.")
                if not matmul:
                    try:
                        cshape = np.broadcast_shapes(cshape, d.cshape)
                    except ValueError as e:
                        raise ValueError(
                            "The cshapes do not match.") from e

        # check type of non signal input
        else:
            if np.asarray(d).dtype.kind not in ["i", "f", "c"]:
                raise ValueError(
                    "Input must be of type Signal, int, float, or complex")
            if (audio_type == (Signal or TimeData)
                    and domain == 'time' and np.asarray(d).dtype.kind == "c"):
                contains_complex = True

    return (sampling_rate, n_samples, fft_norm, times, frequencies, audio_type,
            cshape, contains_complex)


def _get_arithmetic_data(data, domain, cshape, matmul, audio_type,
                         contains_complex):
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
    contains_complex : bool
        Flag which indicates if the operation involves complex-valued pyfar
        audio objects

    Returns
    -------
    data_out : numpy array
        Data in desired domain without any fft normalization if data is a
        Signal. `np.asarray(data)` otherwise.
    """
    if isinstance(data, (Signal, TimeData, FrequencyData)):
        data = data.copy()
        # check if complex casting of any input signal is necessary
        if type(data) is not FrequencyData:
            data.complex = contains_complex
        # get signal in correct domain
        if domain == "time":
            data_out = data.time
        elif domain == "freq":
            if type(data) is Signal:
                data_out = data.freq_raw
            else:
                data_out = data.freq
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
    fft_norm_1 : str
        First fft_norm for matching. Can be ``'none'``, ``'unitary'``,
        ``'amplitude'``, ``'rms'``, ``'power'`` or ``'psd'``
    fft_norm_2 : str
        Second fft_norm for matching. Can be ``'none'``, ``'unitary'``,
        ``'amplitude'``, ``'rms'``, ``'power'`` or ``'psd'``
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

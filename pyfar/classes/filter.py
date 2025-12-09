"""
The following documents the pyfar filter classes. Examples for working with
filter objects are part of the
:doc:`examples gallery<gallery:gallery/interactive/pyfar_filtering>`.
Available filters are shown in the
:doc:`filter types examples<gallery:gallery/interactive/pyfar_filter_types>`
and documented in :py:mod:`pyfar.dsp.filter`.
"""

from abc import ABC, abstractmethod

import deepdiff
import warnings

import numpy as np
import scipy.signal as spsignal
import scipy.linalg as spla

import pyfar as pf
from copy import deepcopy


def _atleast_3d_first_dim(arr):
    arr = np.asarray(arr)
    ndim = np.ndim(arr)

    if ndim < 2:
        arr = np.atleast_2d(arr)
    if ndim < 3:
        return arr[np.newaxis]
    else:
        return arr


def _atleast_4d_first_dim(arr):
    arr = np.asarray(arr)
    ndim = np.ndim(arr)

    if ndim < 3:
        arr = _atleast_3d_first_dim(arr)
    if ndim < 4:
        return arr[np.newaxis]
    else:
        return arr


def _pop_state_from_kwargs(**kwargs):
    kwargs.pop('zi', None)
    warnings.warn(
        "This filter function does not support saving the filter state",
        stacklevel=2)
    return kwargs


def _extend_sos_coefficients(sos, order):
    """
    Extend a set of SOS filter coefficients to match a required filter order
    by adding sections with coefficients resulting in an ideal frequency
    response.

    Parameters
    ----------
    sos : array-like
        The second order section filter coefficients.
    order : int
        The order to which the coefficients are to be extended.

    Returns
    -------
    sos_ext : array-like
        The extended second order section coefficients.

    """
    sos_order = sos.shape[0]
    if sos_order == order:
        return sos
    pad_len = order - sos_order
    sos_ext = np.zeros((pad_len, 6))
    sos_ext[:, 3] = 1.
    sos_ext[:, 0] = 1.

    return np.vstack((sos, sos_ext))


def _repr_string(filter_type, order, n_channels, sampling_rate):
    """Generate repr string for filter objects."""

    ch_str = 'channel' if n_channels == 1 else 'channels'

    if filter_type == "SOS":
        sec_str = 'section' if order == 1 else 'sections'
        representation = (f"SOS filter with {order} {sec_str} and "
                          f"{n_channels} {ch_str} "
                          f"@ {sampling_rate} Hz sampling rate")
    else:
        if order % 10 == 1:
            order_string = 'st'
        elif order % 10 == 2:
            order_string = 'nd'
        elif order % 10 == 3:
            order_string = 'rd'
        else:
            order_string = 'th'

        representation = (f"{order}{order_string} order "
                          f"{filter_type} filter with "
                            f"{n_channels} {ch_str} @ {sampling_rate} "
                            "Hz sampling rate")

    return representation


class _LTISystem(ABC):
    """Abstract base class for LTI systems.

    Parameters
    ----------
    sampling_rate : number
        The sampling rate of the system in Hz.
    state : array, double, optional
        The internal state of the system as an array.
    comment : str
        A comment. The default is ``''``, which initializes an empty string.
    """

    def __init__(self, sampling_rate=None, state=None, comment=""):
        self._state = state
        self._sampling_rate = sampling_rate
        self.comment = comment

    @property
    def sampling_rate(self):
        """Sampling rate of the system in Hz.

        The sampling rate is set upon initialization and cannot be changed
        after the object has been created.
        """
        return self._sampling_rate

    @property
    def state(self):
        """
        The internal state of the system as an array.
        """
        return self._state

    def reset(self):
        """Reset the system state by filling it with zeros."""
        if self._state is not None:
            self._state = np.zeros_like(self._state)
        else:
            self._state = None

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
        """Return a copy of the _LTISystem object."""
        return deepcopy(self)

    def _encode(self):
        """Return dictionary for the encoding."""
        return self.copy().__dict__

    def __eq__(self, other):
        """Check for equality of two objects."""
        return not deepdiff.DeepDiff(self, other)

    @abstractmethod
    def process(self, signal, reset=False):
        """Abstract method that computes the system's forced response."""

    @abstractmethod
    def impulse_response(self, n_samples):
        """Abstract method that computes the system's impulse response."""


class Filter(_LTISystem):
    """Abstract base class for digital filters.

    Parameters
    ----------
    coefficients : array, double
        The filter coefficients as an array.
    sampling_rate : number
        The sampling rate of the filter in Hz.
    state : array, double, optional
        The state of the buffer elements.
    comment : str
        A comment. The default is ``''``, which initializes an empty string.
    """

    def __init__(
            self,
            coefficients=None,
            sampling_rate=None,
            state=None,
            comment=""):
        if coefficients is not None:
            self.coefficients = coefficients
        else:
            self._coefficients = None

        if state is not None:
            if coefficients is None:
                raise ValueError(
                    "Cannot set a state without filter coefficients")
            state = _atleast_3d_first_dim(state)
        super().__init__(sampling_rate=sampling_rate,
                         state=state, comment=comment)

    def init_state(self, state):
        """
        Initialize the buffer elements to pre-defined initial conditions.

        This method is overwritten in the child classes and called after
        setting the state there.
        """
        self._state = state

    @staticmethod
    def _check_state_keyword(state):
        """
        Check the value of the state keyword for 'ini_state' class methods.
        """
        if state not in ['zeros', 'step']:
            raise ValueError(
                f"state is '{state}' but must be 'zeros' or 'step'")

    @property
    def coefficients(self):
        """
        Get and set the coefficients of the filter.

        Refer to the
        :doc:`gallery:gallery/interactive/pyfar_filter_types` for use
        cases.
        """
        return self._coefficients

    @coefficients.setter
    def coefficients(self, value):
        """Coefficients of the filter."""
        self._coefficients = _atleast_3d_first_dim(value)

    @property
    def n_channels(self):
        """The number of channels of the filter."""
        return self._coefficients.shape[0]

    @staticmethod
    @abstractmethod
    def _process(coefficients, data, zi=None):
        """Abstract method that defines the actual filtering process."""

    def process(self, signal, reset=False):
        """Apply the filter to a signal.

        Parameters
        ----------
        signal : Signal
            The data to be filtered as Signal object.
        reset : bool, optional
            If set to ``True``, the filter state will be reset to zeros before
            the filter is applied to the signal. Note that if the filter state
            is ``None``, this option will have no effect. Use ``init_state``
            to initialize a filter with no previously set state. The default
            is ``'False'``.

        Returns
        -------
        filtered : Signal
            A filtered copy of the input signal.
        """
        if not isinstance(signal, pf.Signal):
            raise ValueError("The input needs to be a Signal object.")

        if self.sampling_rate != signal.sampling_rate:
            raise ValueError(
                "The sampling rates of filter and signal do not match")

        if reset is True:
            self.reset()

        # shape of the output signal. if n_channels is 1, it will be squeezed
        # below
        filtered_signal_data = np.zeros(
            (self.n_channels, *signal.time.shape),
            dtype=signal.time.dtype)

        if self.state is not None:
            new_state = np.zeros_like(self._state)
            for idx, (coeff, state) in enumerate(
                    zip(self._coefficients, self._state)):
                filtered_signal_data[idx, ...], new_state[idx, ...] = \
                    self._process(coeff, signal.time, state)
            self._state = new_state
        else:
            for idx, coeff in enumerate(self._coefficients):
                filtered_signal_data[idx, ...] = self._process(
                    coeff, signal.time, zi=None)

        # prepare output signal
        filtered_signal = deepcopy(signal)

        # squeeze first dimension if there is only one filter channel
        if self.n_channels == 1:
            filtered_signal.time = np.squeeze(filtered_signal_data, axis=0)
        else:
            filtered_signal.time = filtered_signal_data

        return filtered_signal

    def impulse_response(self, n_samples):
        """
        Compute or approximate the impulse response of the filter.

        See `impulse_response` methods in derived classes for more details.

        Parameters
        ----------
        n_samples : int, None
            Length in samples for which the impulse response is computed. If
            this is ``None``the length is estimated using
            `minimum_impulse_response_length`.

        Returns
        -------
        impulse_response : Signal
            The impulse response of the filter of with a
            ``cshape = (n_channels, )``, with the channel shape
            :py:func:`~pyfar.Signal.cshape` and the number of filter channels
            :py:func:`~n_channels`.
        """
        # set or check the impulse response length
        minimum_impulse_response_length = int(np.max(
            self.minimum_impulse_response_length()))
        if n_samples is None:
            n_samples = minimum_impulse_response_length
        elif np.any(minimum_impulse_response_length > n_samples):
            warnings.warn(
                ('n_samples should be at least as long as the filter, '
                 f'which is {minimum_impulse_response_length}'), stacklevel=2)

        # track the state (better than copying the entire filter)
        if self.state is not None:
            state = self.state.copy()
            self._state = None
            reset = True
        else:
            reset = False

        # get impulse response
        impulse_response = self.process(pf.signals.impulse(
            n_samples, sampling_rate=self.sampling_rate), reset=reset)

        # reset the state if required
        if reset:
            self._state = state

        return impulse_response

    @classmethod
    def _decode(cls, obj_dict):
        """Decode object based on its respective object dictionary."""
        # initializing this way satisfies FIR, IIR and SOS initialization
        obj = cls(np.zeros((1, 6)), None, None, "")
        obj.__dict__.update(obj_dict)
        return obj


class FilterFIR(Filter):
    """
    Filter object for FIR filters.

    Parameters
    ----------
    coefficients : array, double
        The filter coefficients as an array with dimensions
        (number of channels, number of filter coefficients)
    sampling_rate : number
        The sampling rate of the filter in Hz.
    state : array, double, optional
        The state of the filter from prior information with dimensions
        ``(n_filter_chan, *cshape, order)``, where ``cshape`` is
        the channel shape of the :py:class:`~pyfar.Signal`
        to be filtered.
    comment : str
        A comment. The default is ``''``, which initializes an empty string.
    """

    def __init__(self, coefficients, sampling_rate, state=None, comment=""):

        if state is not None and np.asarray(state).ndim < 3:
            state = _atleast_3d_first_dim(state)
            warnings.warn(
                "The state should be of shape (n_channels, *cshape, order). "
                f"The state has been reshaped to shape=({state.shape}",
                stacklevel=2)

        super().__init__(coefficients, sampling_rate, state, comment)

    @property
    def order(self):
        """The order of the filter."""
        return self._coefficients.shape[-1] - 1

    @property
    def coefficients(self):
        """
        Get and set the coefficients of the filter.

        Refer to the
        :doc:`gallery:gallery/interactive/pyfar_filter_types` for use
        cases.
        """
        # property from Filter is overwritten, because FilterFIR internally
        # also stores a-coefficients easier handling of coefficients across
        # filter classes. The user should only see the b-coefficients, however.
        return self._coefficients[:, 0]

    @coefficients.setter
    def coefficients(self, value):
        """Coefficients of the filter."""

        b = np.atleast_2d(value)
        # add a-coefficients for easier handling across filter classes
        a = np.zeros_like(b)
        a[..., 0] = 1

        coeff = np.stack((b, a), axis=-2)
        self._coefficients = _atleast_3d_first_dim(coeff)

    @property
    def state(self):
        """
        Get or set the filter state.

        The filter state sets the initial condition of the filter and can for
        example be used for block-wise filtering.

        Note that the state can also be initialized with the
        :py:func:`~pyfar.classes.filter.FilterFIR.init_state` method.

        Shape of the state must be  ``(n_filter_channels, *cshape, order)``,
        with ``cshape`` being the channel shape of the
        :py:class:`~pyfar.Signal` to be filtered.
        """
        return self._state

    @state.setter
    def state(self, state):
        """
        Set initial state of FIR filter.

        Parameters
        ----------
        state : array, double
            The state of the filter with dimensions
            ``(n_channels, *cshape, order)``, where ``cshape`` is
            the channel shape of the :py:class:`~pyfar.Signal`
            to be filtered.
        """
        if state is not None:
            state = np.asarray(state)
            if (state.shape[-1] != self.order
                or state.ndim < 3
                or state.shape[0] != self.n_channels):
                raise ValueError(
                    "The state does not match the filter structure. Required "
                    "shape for FilterFIR is (n_channels, *cshape, order).")

        # sets filter state in parent class
        Filter.state.fset(self, state)

    def init_state(self, cshape, state='zeros'):
        """Initialize the buffer elements to pre-defined initial conditions.

        Parameters
        ----------
        cshape : tuple, int
            The channel shape of the :py:class:`~pyfar.Signal`
            which is to be filtered.
        state : str, optional
            The desired state. This can either be ``'zeros'`` which initializes
            an empty filter, or ``'step'`` which constructs the initial
            conditions for step response steady-state. The default is 'zeros'.
        """
        self._check_state_keyword(state)

        new_state = np.zeros((self.n_channels, *cshape, self.order))
        if state == 'step':
            for idx, coeff in enumerate(self._coefficients):
                new_state[idx, ...] = spsignal.lfilter_zi(coeff[0], coeff[1])
        super().init_state(state=new_state)

    def minimum_impulse_response_length(self, unit='samples', tolerance=None):
        r"""
        Get the minimum length of the filter impulse response.

        The length is computed from the last non-zero coefficient per channel.

        Parameters
        ----------
        tolerance : float, optional
            Tolerance for estimating the minimum length of noisy FIR filters.
            The length is estimated by finding the last coefficient with an
            absolute value greater or equal to the absolute maximum of the
            filter coefficients per channel multiplied by `tolerance`. For
            example if ``tolerance = 0.001`` trailing values below
            :math:`20 \log_{10}(0.001)=-60` dB would be ignored in the length
            estimation. The default ``None`` uses the numerical precision
            ``2 * numpy.finfo(float).resolution`` as a strict threshold.
        unit : string, optional
            The unit in which the length is returned. Can be ``'samples'`` or
            ``'s'`` (seconds). The default is ``'samples'``.

        Returns
        -------
        minimum_impulse_response_length : array
            An integer array of shape(n_channels, ) containing the length in
            samples with the number of filter channels :py:func:`~n_channels`.
        """

        # get filter coefficients
        b = self.coefficients.astype(float)

        if tolerance is None:
            thresholds = np.ones(b.shape[0]) * 2 * np.finfo(b.dtype).eps
        else:
            thresholds = np.max(np.abs(b), -1) * tolerance

        # find last entry above tolerance per channel
        above_threshold = b > np.repeat(thresholds[..., None], b.shape[-1], -1)
        estimated_length = np.where(above_threshold)[1].reshape(b.shape[0], -1)
        estimated_length = np.max(estimated_length, -1) + 1

        # convert to desired unit
        if unit == 's':
            estimated_length /= self.sampling_rate
        elif unit == 'samples':
            estimated_length = estimated_length.astype(int)
        else:
            raise ValueError(f"unit is '{unit}' but must be 'samples' or 's'")

        return estimated_length

    def impulse_response(self, n_samples=None):
        """
        Compute the impulse response of the filter.

        Parameters
        ----------
        n_samples : int, optional
            Length in samples for which the impulse response is computed. The
            default is ``None`` in which case ``n_samples`` is computed as the
            maximum value across filter channels returned by
            :py:func:`~FilterFIR.minimum_impulse_response_length`.
            A warning is returned if ``n_samples`` is shorter than the value
            returned by :py:func:`~FilterFIR.minimum_impulse_response_length`.

        Returns
        -------
        impulse_response : Signal
            The impulse response of the filter of with a
            ``cshape = (n_channels, )``, with the channel shape
            :py:func:`~pyfar.Signal.cshape` and the number of filter channels
            :py:func:`~n_channels`.
        """
        return super().impulse_response(n_samples)

    @staticmethod
    def _process(coefficients, data, zi=None):
        """Process a single filter channel.
        This is a hidden static method required for a shared processing
        function in the parent class.

        Parameters
        ----------
        coefficients : array
            Coefficients of the filter channel.
        data : array
            The data to be filtered of shape ``(cshape, n_samples)``.
        zi : array, optional
            The initial filter state of shape ``(cshape, order)`` where
            ``cshape`` is the channel shape of the signal to be filtered.
            The default is ``None``.

        Returns
        -------
        out : array
            The filtered data.
        zf : array
            The final filter state. Only returned if ``zi is not None``.
        """

        b = coefficients[0]
        # broadcast b to match ndim of data for convolution
        b = np.broadcast_to(b, (1, ) * (data.ndim - b.ndim) + b.shape)

        out_full = spsignal.oaconvolve(data, b, mode='full')

        # Ensure that the output has the same shape as the input
        out = out_full[..., 0:data.shape[-1]]

        if zi is not None:

            if zi.shape[0:-1] != data.shape[0:-1]:
                raise ValueError(
                    "The initial state does not match the cshape of "
                    "the signal. Required shape for `state` in "
                    "FilterFIR is (n_channels, *cshape, order).")

            # add initial filter state to the beginning of the output
            out[..., 0:zi.shape[-1]] += zi
            # get new filter state
            zf = out_full[..., data.shape[-1]:]
            return out, zf
        else:
            return out

    def __repr__(self):
        """Representation of the filter object."""
        return _repr_string(
            "FIR", self.order, self.n_channels, self.sampling_rate)


class FilterIIR(Filter):
    """
    Filter object for IIR filters.

    For IIR filters with high orders, second order section IIR filters using
    FilterSOS should be considered.

    Parameters
    ----------
    coefficients : array, double
        The filter coefficients as an array, with shape
        (number of channels, 2, max(number of coefficients in the nominator,
        number of coefficients in the denominator))
    sampling_rate : number
        The sampling rate of the filter in Hz.
    state : array, double, optional
        The state of the filter from prior information with dimensions
        ``(n_filter_chan, *cshape, order)``, where ``cshape`` is
        the channel shape of the :py:class:`~pyfar.Signal`
        to be filtered.
    comment : str
        A comment. The default is ``''``, which initializes an empty string.
    """

    def __init__(self, coefficients, sampling_rate, state=None, comment=""):
        if state is not None and np.asarray(state).ndim < 3:
            state = _atleast_3d_first_dim(state)
            warnings.warn(
                "The state should be of shape (n_channels, *cshape, order). "
                f"The state has been reshaped to shape=({state.shape}",
                stacklevel=2)

        super().__init__(coefficients, sampling_rate, state, comment)

    @property
    def order(self):
        """The order of the filter."""
        return np.max(self._coefficients.shape[-2:]) - 1

    @property
    def state(self):
        """
        Get or set the filter state.

        The filter state sets the initial condition of the filter and can for
        example be used for block-wise filtering.

        Note that the state can also be initialized with the
        :py:func:`~pyfar.classes.filter.FilterIIR.init_state` method.

        Shape of the state must be  ``(n_filter_channels, *cshape, order)``,
        with ``cshape`` being the channel shape of the
        :py:class:`~pyfar.Signal` to be filtered.
        """
        return self._state

    @state.setter
    def state(self, state):
        """
        Set initial state of IIR filter.

        Parameters
        ----------
        state : array, double
            The state of the filter with dimensions
            ``(n_channels, *cshape, order)``, where ``cshape`` is
            the channel shape of the :py:class:`~pyfar.Signal`
            to be filtered.
        """
        if state is not None:
            state = np.asarray(state)
            if (state.shape[-1] != self.order
                or state.ndim < 3
                or state.shape[0] != self.n_channels):
                raise ValueError(
                    "The state does not match the filter structure. Required "
                    "shape for FilterIIR is (n_channels, *cshape, order).")

        # sets filter state in parent class
        Filter.state.fset(self, state)

    def init_state(self, cshape, state='zeros'):
        """Initialize the buffer elements to pre-defined initial conditions.

        Parameters
        ----------
        cshape : tuple, int
            The channel shape of the :py:class:`~pyfar.Signal`
            which is to be filtered.
        state : str, optional
            The desired state. This can either be ``'zeros'`` which initializes
            an empty filter, or ``'step'`` which constructs the initial
            conditions for step response steady-state. The default is 'zeros'.
        """
        self._check_state_keyword(state)

        new_state = np.zeros((self.n_channels, *cshape, self.order))
        if state == 'step':
            for idx, coeff in enumerate(self._coefficients):
                new_state[idx, ...] = spsignal.lfilter_zi(coeff[0], coeff[1])
        return super().init_state(state=new_state)

    def impulse_response(self, n_samples=None):
        """
        Compute the impulse response of the filter.

        Parameters
        ----------
        n_samples : int, optional
            Length in samples for which the impulse response is computed. The
            default is ``None`` in which case ``n_samples`` is computed as the
            maximum value across filter channels returned by
            :py:func:`~FilterIIR.minimum_impulse_response_length`.
            A warning is returned if ``n_samples`` is shorter than the value
            returned by :py:func:`~FilterIIR.minimum_impulse_response_length`.

        Returns
        -------
        impulse_response : Signal
            The impulse response of the filter of with a
            ``cshape = (n_channels, )``, with the channel shape
            :py:func:`~pyfar.Signal.cshape` and the number of filter channels
            :py:func:`~n_channels`.
        """
        return super().impulse_response(n_samples)

    def minimum_impulse_response_length(self, unit='samples', tolerance=5e-5):
        """
        Estimate the minimum length of the filter impulse response.

        The length is estimated from the positions of the poles of the filter.
        The closer the poles are to the unit circle, the longer the estimated
        length.

        Parameters
        ----------
        unit : string, optional
            The unit in which the length is returned. Can be ``'samples'`` or
            ``'s'`` (seconds). The default is ``'samples'``.
        tolerance : float, optional
            Tolerance for the accuracy. Smaller tolerances will result in
            larger impulse response lengths. The default is ``5e-5``.

        Returns
        -------
        minimum_impulse_response_length : array
            An integer array of shape(n_channels, ) containing the length in
            samples with the number of filter channels :py:func:`~n_channels`.
        """
        # This is a direct python port of the parts of Matlab's impzlength that
        # refer to IIR filters

        channels = self.coefficients.shape[0]
        estimated_length = np.zeros(channels)

        for channel in range(channels):

            b = self.coefficients[channel, 0]
            a = self.coefficients[channel, 1]

            # this is an FIR filter
            if self.order == 0 or np.all(a[1:] == 0):
                estimated_length[channel] = np.max(np.nonzero(b)) + 1
                continue

            # This is an IIR filter. Delay of non-recursive part
            estimated_length[channel] = np.min(np.nonzero(b))

            # poles of the transfer function
            poles = np.roots(a)

            if np.any(np.abs(poles) > 1.0001):
                # This is an unstable filter. The closer the outmost pole is to
                # the unit circle, the longer the impulse response.
                estimated_length[channel] += \
                    6 / np.log10(np.max(np.abs(poles[np.abs(poles) > 1])))
            else:
                # This is a stable filter. Minimum height is 1e-5 of original
                # amplitude
                idx = np.abs(poles - 1) < 1e-5
                poles[idx] = -poles[idx]

                # poles on and close to unit circle
                idx_oscillation = np.argwhere(np.abs(np.abs(poles)-1) < 1e-5)
                # poles further away from unit circle
                idx_damped = np.argwhere(np.abs(np.abs(poles)-1) >= 1e-5)

                if len(idx_oscillation) == len(poles):
                    # pure oscillation
                    estimated_length[channel] += \
                        5 * np.max(2 * np.pi / np.abs(np.angle(poles)))
                elif len(idx_damped) == len(poles):
                    # no oscillation
                    idx = np.argmax(np.abs(poles))
                    pole_multiplicity = self._pole_multiplicity(poles, idx)
                    estimated_length[channel] += \
                        pole_multiplicity * np.log10(tolerance) / \
                        np.log10(np.abs(poles[idx]))
                else:
                    # mixture of both
                    periods = 5 * np.max(2 * np.pi / \
                        np.abs(np.angle(poles[idx_oscillation[0]])))
                    poles_damped = poles[idx_damped[0]]
                    idx = np.argmax(np.abs(poles_damped))
                    multiplicity = self._pole_multiplicity(poles_damped, idx)
                    # catch runtime warning for poles in the origin
                    if np.abs(poles_damped[idx]) > 0:
                        multiplicity_factor = np.log10(tolerance) / \
                            np.log10(np.abs(poles_damped[idx]))
                    else:
                        multiplicity_factor = 0

                    estimated_length[channel] += np.maximum(
                        periods,
                        multiplicity * multiplicity_factor)

            estimated_length[channel] = np.maximum(
                len(a) + len(b) - 1, estimated_length[channel])

        if unit == 's':
            estimated_length /= self.sampling_rate
        elif unit == 'samples':
            estimated_length = estimated_length.astype(int)
        else:
            raise ValueError(f"unit is '{unit}' but must be 'samples' or 's'")

        return estimated_length

    @staticmethod
    def _pole_multiplicity(poles, index, tolerance=.001):
        """
        Find multiplicity of a pole.

        Required for minimum_impulse_response_length.
        """

        if np.all(poles != 0):
            tolerance *= np.abs(poles[index])

        return np.sum(np.abs(poles - poles[index]) < tolerance)

    @staticmethod
    def _process(coefficients, data, zi=None):
        """Process a single filter channel.
        This is a hidden static method required for a shared processing
        function in the parent class.
        """
        if zi is not None and zi.shape[0:-1] != data.shape[0:-1]:
            raise ValueError("The initial state does not match the cshape of "
                             "the signal. Required shape for `state` in "
                             "FilterIIR is (n_channels, *cshape, order).")
        return spsignal.lfilter(coefficients[0], coefficients[1], data, zi=zi)

    def __repr__(self):
        """Representation of the filter object."""
        return _repr_string(
            "IIR", self.order, self.n_channels, self.sampling_rate)


class FilterSOS(Filter):
    """
    Filter object for IIR filters as second order sections (SOS).

    Parameters
    ----------
    coefficients : array, double
        The filter coefficients as an array with dimensions
        ``(n_filter_chan, n_sections, 6)`` The first three values of
        a section provide the numerator coefficients, the last three values
        the denominator coefficients, e.g,
        ``[[[ b[0], b[1], b[2], a[0], a[1], a[2] ]]]`` for a single channel
        SOS filter with one section.
    sampling_rate : number
        The sampling rate of the filter in Hz.
    state : array, double, optional
        The state of the filter from prior information with dimensions
        ``(n_filter_chan, *cshape, n_sections, 2)``, where ``cshape`` is
        the channel shape of the :py:class:`~pyfar.Signal`
        to be filtered.
    comment : str
        A comment. The default is ``''``, which initializes an empty string.
    """

    def __init__(self, coefficients, sampling_rate, state=None, comment=""):
        if state is not None and np.asarray(state).ndim < 4:
            state = _atleast_4d_first_dim(state)
            warnings.warn(
                "The state should be of shape (n_channels, *cshape, n_sections"
                f", 2). The state has been reshaped to shape=({state.shape}",
                stacklevel=2)

        super().__init__(coefficients, sampling_rate, state, comment)

    @Filter.coefficients.setter
    def coefficients(self, value):
        """Coefficients of the filter."""

        coeff = _atleast_3d_first_dim(value)
        if coeff.shape[-1] != 6:
            raise ValueError(
                "The coefficients are not in line with a second order",
                "section filter structure.")

        self._coefficients = coeff

    @property
    def order(self):
        """The order of the filter.
        This is always twice the number of sections.
        """
        return 2*self.n_sections

    @property
    def n_sections(self):
        """The number of sections."""
        return self._coefficients.shape[-2]

    @property
    def state(self):
        """
        Get or set the filter state.

        The filter state sets the initial condition of the filter and can for
        example be used for block-wise filtering.

        Note that the state can also be initialized with the
        :py:func:`~pyfar.classes.filter.FilterSOS.init_state` method.

        Shape of the state must be
        ``(n_filter_channels, *cshape, n_sections, 2))``,
        with ``cshape`` being the channel shape of the
        :py:class:`~pyfar.Signal` to be filtered.
        """
        return self._state

    @state.setter
    def state(self, state):
        """
        Set initial state of SOS filter.

        Parameters
        ----------
        state : array, double
            The state of the filter with dimensions
            ``(n_channels, *cshape, n_sections, 2)``, where ``cshape`` is
            the channel shape of the :py:class:`~pyfar.Signal`
            to be filtered.
        """
        if state is not None:
            state = np.asarray(state)
            if (state.shape[0] != self.n_channels
                or state.shape[-1] != 2
                or state.shape[-2] != self.n_sections
                or state.ndim < 4):
                raise ValueError(
                    "The state does not match the filter structure. Required "
                    "shape for FilterSOS is (n_channels, *cshape, "
                    "n_sections, 2).")

        # sets filter state in parent class
        Filter.state.fset(self, state)

    def init_state(self, cshape, state='zeros'):
        """Initialize the buffer elements to pre-defined initial conditions.

        Parameters
        ----------
        cshape : tuple, int
            The channel shape of the :py:class:`~pyfar.Signal`
            which is to be filtered.
        state : str, optional
            The desired state. This can either be ``'zeros'`` which initializes
            an empty filter, or ``'step'`` which constructs the initial
            conditions for step response steady-state. The default is 'zeros'.
        """
        self._check_state_keyword(state)

        new_state = np.zeros((self.n_channels, *cshape, self.n_sections, 2))
        if state == 'step':
            for idx, coeff in enumerate(self._coefficients):
                new_state[idx, ...] = spsignal.sosfilt_zi(coeff)
        return super().init_state(state=new_state)

    def impulse_response(self, n_samples=None):
        """
        Approximate the infinite impulse response of the filter by a finite
        impulse response.

        Note that the number of samples must be sufficiently long for
        `impulse_response` to be a good approximation of the theoretically
        infinitely long impulse response of the filter.

        Parameters
        ----------
        n_samples : int
            Length in samples for which the impulse response is computed. The
            default is ``None`` in which case ``n_samples`` is computed as the
            maximum value across filter channels returned by
            :py:func:`~FilterSOS.minimum_impulse_response_length`.
            A warning is returned if ``n_samples`` is shorter than the value
            returned by :py:func:`~FilterSOS.minimum_impulse_response_length`.

        Returns
        -------
        impulse_response : Signal
            The impulse response of the filter of with a
            ``cshape = (n_channels, )``, with the channel shape
            :py:func:`~pyfar.Signal.cshape` and the number of filter channels
            :py:func:`~n_channels`.
        """
        return super().impulse_response(n_samples)

    def minimum_impulse_response_length(self, unit='samples', tolerance=5e-5):
        """
        Estimate the minimum length of the filter impulse response.

        The length is estimated separately per second order section using
        :py:func:`~FilterIIR.minimum_impulse_response_length`. The final length
        is given by the length of the longest section.

        Parameters
        ----------
        unit : string, optional
            The unit in which the length is returned. Can be ``'samples'`` or
            ``'s'`` (seconds). The default is ``'samples'``.
        tolerance : float, optional
            Tolerance for the accuracy. Smaller tolerances will result in
            larger impulse response lengths. The default is ``5e-5``.

        Returns
        -------
        minimum_impulse_response_length : array
            An integer array of shape(n_channels, ) containing the length in
            samples with the number of filter channels :py:func:`~n_channels`.
        """
        # This is a direct python port of the parts of Matlab's impzlength that
        # refer to SOS filters

        channels = self.coefficients.shape[0]
        sections = self.coefficients.shape[1]
        estimated_length = np.zeros(channels)

        for channel in range(channels):

            # initialize length for FIR and IIR sections
            # Note: Matlab uses 1 for initialization, which does not make sense
            #       for FIR sections. Using 0 should be better there and does
            #       not change anything for IIR sections
            length_fir = 0
            length_iir = 0

            for section in range(sections):

                # estimate length of each section and channel
                b = self.coefficients[channel, section, :3]
                a = self.coefficients[channel, section, 3:]

                if np.all(a[1:] == 0):
                    # FIR sections: accumulate FIR length
                    length_fir += np.max(np.nonzero(b)) + 1
                else:
                    # IIR section: track maximum IIR length
                    # (assuming this dominates the impulse response length)
                    filter_iir = FilterIIR([b, a], self.sampling_rate)

                    length_iir = np.maximum(
                        length_iir,
                        filter_iir.minimum_impulse_response_length(
                            tolerance=tolerance)[0])

            # use maximum of FIR and IIR length for final estimate
            estimated_length[channel] = np.maximum(length_fir, length_iir)

        if unit == 's':
            estimated_length /= self.sampling_rate
        elif unit == 'samples':
            estimated_length = estimated_length.astype(int)
        else:
            raise ValueError(f"unit is '{unit}' but must be 'samples' or 's'")

        return estimated_length

    @staticmethod
    def _process(sos, data, zi=None):
        """Process a single filter channel.
        This is a hidden static method required for a shared processing
        function in the parent class.
        """
        if zi is not None and zi.shape[0:-2] != data.shape[0:-1]:
            raise ValueError("The initial state does not match the cshape of "
                             "the signal. Required shape for `state` in "
                             "FilterSOS is (n_channels, *cshape, n_sections,"
                             " 2).")
        if zi is not None:
            zi = zi.transpose(1, 0, 2)
        res = spsignal.sosfilt(sos, data, zi=zi, axis=-1)
        if zi is not None:
            zi = res[1].transpose(1, 0, 2)
            return res[0], zi
        else:
            return res

    def __repr__(self):
        """Representation of the filter object."""
        return _repr_string(
            "SOS", self.n_sections, self.n_channels, self.sampling_rate)


class StateSpaceModel(_LTISystem):
    """
    Class for discrete-time state-space models.

    A state-space model is defined by the matrices :math:`A`, :math:`B`,
    :math:`C`, and :math:`D`. Contrary to an impulse response or filter model,
    a state-space model can represent multi-input multi-output (MIMO) linear
    time-invariant (LTI) systems in a numerically stable and efficient way. The
    system is described by the following equations (discrete-time):

    .. math:
        x_{t+1} = A x_t + B u_t
        y_t   = C x_t + D u_t,

    where :math:`x_t` is the state, :math:`u` the input, and :math:`y` the
    output at time step :math:`t`.

    The matrix :math:`A` is the state matrix, defining the internal dynamics of
    the system, i.e. how the internal state evolves on its own; :math:`B` is
    the input matrix, defining the action of the input :math: `u` on the state;
    :math:`C` is the output matrix, defining the action of the state :math:`x`
    on the output :math:`y`; and :math:`D` is the feedthrough matrix, defining
    the direct action of the input :math:`u` on the output :math:`y`.

    Note that, unlike an impulse response or transfer function representation,
    the state-space representation is not unique. Different state-space models
    can represent the same input-output
    behaviour.

    For a short introduction, see: https://ccrma.stanford.edu/~jos/StateSpace/

    Parameters
    ----------
    A : array
        The state matrix :math:`A` with dimensions ``(order, order)``.
    B : array
        The input matrix :math:`B` with dimensions ``(order, n_inputs)``.
    C : array
        The output matrix :math:`C` with dimensions ``(n_outputs, order)``.
    D : array, optional
        The feedthrough matrix :math:`D` with dimensions ``(n_outputs,
        n_inputs)``. The default is ``None``, which initializes an all-zero
        matrix.
    sampling_rate : number, optional
        The sampling rate of the system in Hz. The default is ``None``.
    state : array, double, optional
        The initial state of the system.
    dtype : np.dtype, optional
        The data type of the system matrices. Can be used to set the precision
        of the response calculation. If ``None``, ``np.promote_types`` is used.
    comment : str, optional
        A comment. The default is ``''``.
    """

    def __init__(self, A, B, C, D=None, sampling_rate=None, state=None,
            dtype=None, comment=""):
        D = np.zeros((C.shape[0], B.shape[1])) if D is None else D
        assert all(
            isinstance(M, np.ndarray) and (M.ndim == 2) for M in (A, B, C, D)
        )
        assert A.shape[1] == A.shape[0], "A needs to be square."
        assert B.shape[0] == A.shape[0], (
            f"B needs to be of shape ({A.shape[0]}, m)."
        )
        assert C.shape[1] == A.shape[0], (
            f"C needs to be of shape (p, {A.shape[0]})."
        )
        assert D.shape == (C.shape[0], B.shape[1]), (
            f"D needs to be of shape ({C.shape[0], B.shape[1]})."
        )
        dtype = np.result_type(A, B, C, D) if dtype is None else dtype
        A = A.astype(dtype, order='F')
        B = B.astype(dtype, order='F')
        C = C.astype(dtype, order='F')
        D = D.astype(dtype, order='F')
        super().__init__(
            sampling_rate=sampling_rate, state=state, comment=comment,
        )
        self._A, self._B, self._C, self._D, self._dtype = A, B, C, D, dtype

    @property
    def A(self):
        """Get the state matrix :math:`A` (internal dynamics of the system)."""
        return self._A

    @property
    def B(self):
        """Get the input matrix :math:`B` (input-to-state mapping)."""
        return self._B

    @property
    def C(self):
        """Get the output matrix :math:`C` (state-to-output mapping)."""
        return self._C

    @property
    def D(self):
        """Get the feedthrough matrix :math:`D` (instant in-out dynamics)."""
        return self._D

    @property
    def dtype(self):
        """Get the data type of the system matrices."""
        return self._dtype

    @property
    def n_inputs(self):
        """Get the number of inputs :math:`m` of the system."""
        return self.B.shape[1]

    @property
    def n_outputs(self):
        """Get the number of outputs :math:`p` of the system."""
        return self.C.shape[0]

    @property
    def order(self):
        """Get the order :math:`n` of the state-space system."""
        return self.A.shape[0]

    @property
    def state(self):
        """Get or set the internal state :math:`x` of the system.

        The state can be used to set the initial condition of the system and to
        save the system's state after processing a signal.

        Parameters
        ----------
        state : array, self.dtype
            The internal state of the system with dimensions ``(order,)``.
        """
        return self._state

    @state.setter
    def state(self, state):
        assert state.shape == (self.order,)
        assert state.dtype == self.dtype
        self._state = state

    def init_state(self):
        """Initialize the internal state :math:`x` of the system to zero."""
        self._state = np.zeros(self.order, self.dtype)

    def process(self, signal, reset=False):
        """Apply the state-space model to a signal.

        Parameters
        ----------
        signal : Signal
            The input signal to be processed.
        reset : bool, optional
            If ``True``, the internal state of the system is reset before
            processing the signal. The default is ``False``.

        Returns
        -------
        out : Signal
            The processed output signal.
        """
        assert signal.sampling_rate == self.sampling_rate, (
            "The sampling rates of the signal and the state-space model"
            "need to match."
        )
        if signal.n_samples == 1:
            u = np.squeeze(signal.time)[:, np.newaxis]
        else:
            u = np.atleast_2d(signal.time.squeeze())
        assert u.shape[0] == self.n_inputs, (
            f"The input signal ({u.shape}) needs to be compatible with the"
            f" number of inputs to the state-space system ({self.n_inputs})."
        )
        if self.state is None or reset:
            self.init_state()

        return pf.Signal(self._process(u), self.sampling_rate)

    def _process(self, u):
        u = np.asfortranarray(u)
        y = np.zeros((self.n_outputs, u.shape[1]), self.dtype, order='F')
        gemv = spla.get_blas_funcs("gemv", dtype=self.dtype)[0]
        for i in range(u.shape[1]):
            y[:, i] = gemv(
                1.,
                self._C,
                self._state,
                beta=1,
                y=gemv(1., self._D, u[:, i], beta=0, y=y[:, i]),
            )
            self._state = gemv(
                1.,
                self._B,
                u[:, i],
                beta=1,
                y=gemv(1., self._A, self._state, beta=0, y=self._state),
            )
        return y

    def impulse_response(self, n_samples):
        """Compute the impulse response of the system.

        Parameters
        ----------
        n_samples : int, None
            The number of samples up to which to compute the impulse response.

        Returns
        -------
        impulse_response : Signal
            The impulse response of the state-space system with ``cshape =
            (n_inputs, n_outputs)``. Note that, somewhat unintuitively, the
            mathematical definition is different and would be ``(n_outputs,
            n_inputs)``.
        """
        y = np.zeros((self.n_inputs, self.n_outputs, n_samples), self.dtype)
        y[..., 0] = self.D.T
        x = self.B
        for i in range(1, n_samples):
            y[..., i] = (self.C @ x).T
            x = self.A @ x
        return pf.Signal(y, self.sampling_rate)

    @classmethod
    def _decode(cls, obj_dict):
        """Decode object based on its respective object dictionary."""
        obj = cls(*np.zeros((3, 1, 1)), None, None, "")
        obj.__dict__.update(obj_dict)
        return obj

    def __repr__(self):
        """Representation of the state-space model."""
        return (
            f"Order {self.order} state-space model with {self.n_inputs} inputs"
            f" and {self.n_outputs} outputs"
            f" @ {self.sampling_rate} Hz sampling rate ({self.dtype})."
        )

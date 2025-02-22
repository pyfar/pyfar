"""
The following documents the pyfar filter classes. Examples for working with
filter objects are part of the
:doc:`examples gallery<gallery:gallery/interactive/pyfar_filtering>`.
Available filters are shown in the
:doc:`filter types examples<gallery:gallery/interactive/pyfar_filter_types>`
and documented in :py:mod:`pyfar.dsp.filter`.
"""
import deepdiff
import warnings

import numpy as np
import scipy.signal as spsignal

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


class Filter(object):
    """
    Container class for digital filters.
    This is an abstract class method, only used for the shared processing
    method used for the application of a filter on a signal.
    """

    def __init__(
            self,
            coefficients=None,
            sampling_rate=None,
            state=None,
            comment=""):
        """
        Initialize a general Filter object.

        Parameters
        ----------
        coefficients : array, double
            The filter coefficients as an array.
        sampling_rate : number
            The sampling rate of the filter in Hz.
        state : array, double, optional
            The state of the buffer elements.
        comment : str
            A comment. The default is ``''``, which initializes an empty
            string.

        Returns
        -------
        Filter
            The filter object.

        """

        if coefficients is not None:
            self.coefficients = coefficients
        else:
            self._coefficients = None

        if state is not None:
            if coefficients is None:
                raise ValueError(
                    "Cannot set a state without filter coefficients")
            state = _atleast_3d_first_dim(state)
            self._initialized = True
        else:
            self._initialized = False

        self._state = state
        self._sampling_rate = sampling_rate
        self.comment = comment

    def init_state(self, state):
        """
        Initialize the buffer elements to pre-defined initial conditions.

        This method is overwritten in the child classes and called after
        setting the state there.
        """
        self._state = state
        self._initialized = True

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
    def sampling_rate(self):
        """Sampling rate of the filter in Hz. The sampling rate is set upon
        initialization and cannot be changed after the object has been created.
        """
        return self._sampling_rate

    @property
    def n_channels(self):
        """The number of channels of the filter."""
        return self._coefficients.shape[0]

    @property
    def state(self):
        """
        The current state of the filter as an array with dimensions
        corresponding to the order of the filter and number of filter channels.
        """
        return self._state

    @staticmethod
    def _process(coefficients, data, zi=None):
        raise NotImplementedError("Abstract class method.")

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
        n_samples : int
            Length in samples for which the impulse response is computed.

        Returns
        -------
        impulse_response : Signal
            The impulse response of the filter.
        """
        # set or check the impulse response length
        impulse_response_length = int(np.max(self.impulse_response_length()))
        if n_samples is None:
            n_samples = impulse_response_length
        elif impulse_response_length > n_samples:
            warnings.warn(
                ('n_samples should be at least as long as the filter, '
                 f'which is {impulse_response_length}'), stacklevel=2)

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

    def reset(self):
        """Reset the filter state by filling it with zeros."""
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
        """Return a copy of the Filter object."""
        return deepcopy(self)

    def _encode(self):
        """Return dictionary for the encoding."""
        return self.copy().__dict__

    @classmethod
    def _decode(cls, obj_dict):
        """Decode object based on its respective object dictionary."""
        # initializing this way satisfies FIR, IIR and SOS initialization
        obj = cls(np.zeros((1, 6)), None)
        obj.__dict__.update(obj_dict)
        return obj

    def __eq__(self, other):
        """Check for equality of two objects."""
        return not deepdiff.DeepDiff(self, other)


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
            A comment. The default is ``''``, which initializes an empty
            string.

    Returns
    -------
    FilterFIR
        The FIR filter object.
    """

    def __init__(self, coefficients, sampling_rate, state=None, comment=""):

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

    def impulse_response_length(self):
        """
        Get the length of filter impulse response in samples.

        The length is computed from the last non-zero coefficient per channel.

        Returns
        -------
        impulse_response_length : array
            An integer array of shape (C, ) containing the length in samples
            where `C` denotes the number of channels of the filter.
        """

        # get filter coefficients
        b = self.coefficients

        # find last non-zero entry per channel
        estimated_length = np.apply_along_axis(
            lambda b: np.max(np.nonzero(b)), axis=-1, arr=b) + 1

        # restore input data shape
        estimated_length = np.reshape(estimated_length, b.shape[:-1])

        return estimated_length.astype(int)

    def impulse_response(self, n_samples=None):
        """
        Compute the impulse response of the filter.

        Parameters
        ----------
        n_samples : int, optional
            Length in samples for which the impulse response is computed. The
            default is ``None`` in which case ``n_samples`` is computed as the
            maximum value returned by :py:func:`~impulse_response_length`.
            A warning is returned if ``n_samples`` is too short to compute the
            entire impulse response.

        Returns
        -------
        impulse_response : Signal
            The impulse response of the filter of with a ``cshape = (C, )``
            where `C` denotes the number of channels of the filter.
        """
        return super().impulse_response(n_samples)

    @staticmethod
    def _process(coefficients, data, zi=None):
        """Process a single filter channel.
        This is a hidden static method required for a shared processing
        function in the parent class.
        """
        return spsignal.lfilter(coefficients[0], 1, data, zi=zi)

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
            A comment. The default is ``''``, which initializes an empty
            string.

    Returns
    -------
    FilterIIR
        The IIR filter object.
    """

    def __init__(self, coefficients, sampling_rate, state=None, comment=""):

        super().__init__(coefficients, sampling_rate, state, comment)

    @property
    def order(self):
        """The order of the filter."""
        return np.max(self._coefficients.shape[-2:]) - 1

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
            maximum value returned by :py:func:`~impulse_response_length`.
            A warning is returned if ``n_samples`` is too short to approximate
            the theoretically infinitely long impulse response.

        Returns
        -------
        impulse_response : Signal
            The impulse response of the filter of with a ``cshape = (C, )``
            where `C` denotes the number of channels of the filter.
        """
        return super().impulse_response(n_samples)

    def impulse_response_length(self, tolerance=5e-5):
        """
        Get the estimated length of filter impulse response in samples.

        The length is estimated from the positions of the poles of the filter.
        The closer the poles are to the unit circle, the longer the estimated
        length.

        Parameters
        ----------
        tolerance : float, optional
            Tolerance for the accuracy. Smaller tolerances will result in
            larger impulse response lengths. The default is ``5e-5``.

        Returns
        -------
        impulse_response_length : array
            An integer array of shape (C, ) containing the length in samples
            where `C` denotes the number of channels of the filter.
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

                # poles on and close to unit circls
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
                    estimated_length[channel] += np.maximum(
                        periods,
                        multiplicity * np.log10(tolerance) / \
                            np.log10(np.abs(poles_damped[idx])))

            estimated_length[channel] = np.maximum(
                len(a) + len(b) - 1, estimated_length[channel])

        return estimated_length.astype(int)

    @staticmethod
    def _pole_multiplicity(poles, index, tolerance=.001):
        """
        Find multiplicity of a pole.

        Required for impulse_response_length.
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
            A comment. The default is ``''``, which initializes an emptry
            string.

    Returns
    -------
    FilterSOS
        The SOS filter object.
    """

    def __init__(self, coefficients, sampling_rate, state=None, comment=""):

        if state is not None:
            state = _atleast_4d_first_dim(state)

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
            maximum value returned by :py:func:`~impulse_response_length`.
            A warning is returned if ``n_samples`` is too short to approximate
            the theoretically infinitely long impulse response.

        Returns
        -------
        impulse_response : Signal
            The impulse response of the filter.
        """
        return super().impulse_response(n_samples)

    def impulse_response_length(self, tolerance=5e-5):
        """
        Get the estimated length of filter impulse response in samples.

        The length is estimated separately per second order section using
        :py:func:`FilterIIR.impulse_response_length`. The final length is given
        by the length of the longest section.

        Parameters
        ----------
        tolerance : float, optional
            Tolerance for the accuracy. Smaller tolerances will result in
            larger impulse response lengths. The default is ``5e-5``.

        Returns
        -------
        impulse_response_length : array
            An integer array of shape (C, ) containing the length in samples
            where `C` denotes the number of channels of the filter.
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
                        filter_iir.impulse_response_length(tolerance)[0])

            # use maximum of FIR and IIR length for final estimate
            estimated_length[channel] = np.maximum(length_fir, length_iir)

        return estimated_length.astype(int)

    @staticmethod
    def _process(sos, data, zi=None):
        """Process a single filter channel.
        This is a hidden static method required for a shared processing
        function in the parent class.
        """
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

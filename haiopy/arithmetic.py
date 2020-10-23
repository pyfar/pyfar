"""Provide arithmethic operations for Signals and array like data."""

# TODO: Account for signal_type in `_get_arithmetic_data`
# TODO: return comment in `_assert_match_for_arithmetic`

import numpy as np
from haiopy.haiopy import Signal


def add(data: tuple, domain='freq'):
    return _arithmetic(data, domain, 'add')


def subtract(data: tuple, domain='freq'):
    return _arithmetic(data, domain, 'subtract')


def multiply(data: tuple, domain='freq'):
    return _arithmetic(data, domain, 'multiply')


def divide(data: tuple, domain='freq'):
    return _arithmetic(data, domain, 'divide')


def power(data: tuple, domain='freq'):
    return _arithmetic(data, domain, 'power')


def _arithmetic(data: tuple, domain: str, operation: str):

    # check input and obtain meta data of new signal
    sampling_rate, n_samples, signal_type = \
        _assert_match_for_arithmetic(data, domain)

    result = _get_arithmetic_data(data[0], domain, signal_type)

    for d in range(1, len(data)):
        if operation == "add":
            result = _add(
                result, _get_arithmetic_data(data[d], domain, signal_type))
        elif operation == "subtract":
            result = _subtract(
                result, _get_arithmetic_data(data[d], domain, signal_type))
        elif operation == "multiply":
            result = _multiply(
                result, _get_arithmetic_data(data[d], domain, signal_type))
        elif operation == "divide":
            result = _divide(
                result, _get_arithmetic_data(data[d], domain, signal_type))
        elif operation == "power":
            result = _power(
                result, _get_arithmetic_data(data[d], domain, signal_type))

    if sampling_rate is not None:
        result = Signal(result, sampling_rate, n_samples, domain, signal_type)

    return result


def _assert_match_for_arithmetic(data: tuple, domain: str):
    """Check if type and meta data of input is fine for arithmetic operations.

    Check if sampling rate and number of samples agree if multiple signals are
    provided. Check if arrays are numeric. Check if a power signal is contained
    in the input.

    Input:
    data : tuple
        Can contain signal and array like data
    domain : str
        Domain in which the arithmetic operation should be performed. 'time' or
        'freq'.

    Returns
    -------
    sampling_rate : number, None
        sampling rate of the signals. None, if no signal is contained in `data`
    n_samples : number, None
        number of samples of the signals. None, if no signal is contained in
        `data`
    signal_type : str, None
        'energy' if all signaly are of type energy. 'power' if any power signal
        is contained in `data`. None if no signal is contained in `data`

    """

    # we need at least two signals
    if not isinstance(data, tuple):
        raise ValueError("Input argument 'data' must be a tuple.")

    # check validity of domain
    if domain not in ['time', 'freq']:
        raise ValueError(f"domain must be time or freq but is {domain}.")

    # check input types and meta data
    sampling_rate = None
    n_samples = None
    signal_type = None
    for d in data:
        # check or store meta data of signals
        if isinstance(d, Signal):
            if sampling_rate is None:
                sampling_rate = d.sampling_rate
                n_samples = d.n_samples
                signal_type = d.signal_type
            else:
                if sampling_rate != d.sampling_rate:
                    raise ValueError("The sampling rates do not match.")
                if n_samples != d.n_samples:
                    raise ValueError("The number of samples does not match.")
            # if there is a power signal, the returned signal will be a power
            # signal
            if d.signal_type == "power":
                signal_type = "power"
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

    return sampling_rate, n_samples, signal_type


def _get_arithmetic_data(data, domain, signal_type):
    """Return data for arithmetic operations.

    Returns
    -------
    data_out : numpy array
        signal data as np.array according to `domain` and `signal_type` if data
        is a Signal and np.array containing `data` otherwise.
    """
    if isinstance(data, Signal):
        # get signal in correct domain
        if domain == "time":
            data_out = data.time.copy()
        elif domain == "freq":
            data_out = data.freq.copy()
        else:
            raise ValueError(
                f"domain must be 'time' or 'freq' but found {domain}")
        # change signal type
        if domain == "freq" and signal_type != data.signal_type:
            # TODO: adjust signal type after normalization is available from
            #       fft module
            data_out = data_out
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

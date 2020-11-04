"""Provide arithmethic operations for Signals and array like data."""

import numpy as np
from typing import Callable
from pyfar.signal import Signal
from pyfar.fft import normalization


def add(data: tuple, domain='freq'):
    return _arithmetic(data, domain, _add)


def subtract(data: tuple, domain='freq'):
    return _arithmetic(data, domain, _subtract)


def multiply(data: tuple, domain='freq'):
    return _arithmetic(data, domain, _multiply)


def divide(data: tuple, domain='freq'):
    return _arithmetic(data, domain, _divide)


def power(data: tuple, domain='freq'):
    return _arithmetic(data, domain, _power)


def _arithmetic(data: tuple, domain: str, operation: Callable):

    # check input and obtain meta data of new signal
    sampling_rate, n_samples, signal_type, fft_norm = \
        _assert_match_for_arithmetic(data, domain)

    result = _get_arithmetic_data(
        data[0], n_samples, domain, signal_type, fft_norm)

    for d in range(1, len(data)):
        result = operation(
            result, _get_arithmetic_data(
                data[d], n_samples, domain, signal_type, fft_norm))

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
    fft_norm : str, None
        fft_norm of the first signal in `data`. None if no signal is contained
        in `data`.

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
    fft_norm = None
    for d in data:
        # check or store meta data of signals
        if isinstance(d, Signal):
            if sampling_rate is None:
                sampling_rate = d.sampling_rate
                n_samples = d.n_samples
                signal_type = d.signal_type
                fft_norm = d.fft_norm
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

    return sampling_rate, n_samples, signal_type, fft_norm


def _get_arithmetic_data(data, n_samples, domain, signal_type, fft_norm):
    """
    Return signal data in desired domain, signal_type and fft_norm.

    Parameters
    ----------
    data : Signal, array like, number
        input data
    n_samples :
        Number of samples of data if data is a Signal (required for fft
        normalization).
    domain, signal_type, fft_norm : attributes
        Attributes in which data_out has to be returned

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

            if signal_type == 'energy' and data.signal_type == 'power':
                # change from 'power' to 'energy' (remove current fft_norm)
                data_out = normalization(
                    data_out, n_samples, data.sampling_rate,
                    'power', data.fft_norm, inverse=True)
            elif signal_type == 'power' and data.signal_type == 'energy':
                # change from 'energy' to 'power' (apply desired fft_norm)
                data_out = normalization(
                    data_out, n_samples, data.sampling_rate,
                    'power', fft_norm, inverse=False)
            elif fft_norm != data.fft_norm:
                # revert current and apply desired fft normalization
                data_out = normalization(
                    data_out, n_samples, data.sampling_rate,
                    'power', data.fft_norm, inverse=True)
                data_out = normalization(
                    data_out, n_samples, data.sampling_rate,
                    'power', fft_norm, inverse=False)
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

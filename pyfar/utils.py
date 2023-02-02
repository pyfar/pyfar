import pyfar as pf
import numpy as np


def broadcast_cshape(signal, cshape):
    """
    Broadcast a signal to a certain cshape.

    The :py:mod:`cshape <pyfar._concepts.audio_classes>` of the signal is
    broadcasted following the `numpy broadcasting rules
    <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_

    Parameters
    ----------
    signal : Signal, TimeData, FrequencyData
        The signal to be broadcasted.
    cshape : tuple
        The cshape to which `signal` is broadcasted.

    Returns
    -------
    signal : Signal, TimeData, FrequencyData
        Broadcasted copy of the input signal
    """

    if not isinstance(signal, (pf.Signal, pf.TimeData, pf.FrequencyData)):
        raise TypeError("Input data must be a pyfar audio object")

    signal = signal.copy()
    signal._data = np.broadcast_to(
        signal._data, cshape + (signal._data.shape[-1], ))
    return signal


def broadcast_cshapes(signals, cshape=None, ignore_axis=None):
    """
    Broadcast multiple signals to a common cshape.

    The :py:mod:`cshape <pyfar._concepts.audio_classes>` of the signals are
    broadcasted following the `numpy broadcasting rules
    <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_

    Parameters
    ----------
    signals : tuple of Signal, TimeData, FrequencyData
        The signals to be broadcasted in a tuple.
    cshape : tuple, optional
        The cshape to which the signals are broadcasted. If `cshape` is
        ``None`` it is determined from the cshapes of the input signals using
        ``numpy.broadcast_shapes``. The default is ``None``.

    Returns
    -------
    signals : tuple of Signal, TimeData, FrequencyData
        The broadcasted copies of the input signals in a tuple.
    """

    for signal in signals:
        if not isinstance(signal, (pf.Signal, pf.TimeData, pf.FrequencyData)):
            raise TypeError("All input data must be pyfar audio objects")

    if cshape is None:
        if ignore_axis is not None:
            data_shapes = [s.cshape for s in signals]
            cshape = np.broadcast_shapes(*np.delete(data_shapes, ignore_axis,
                                         axis=-1))
            if ignore_axis in (-1, len(cshape)):
                cshape = [np.append(cshape, data_shapes[idx][ignore_axis])
                          for idx, s in enumerate(signals)]
            else:
                insert_axis = ignore_axis+1 if ignore_axis < 0 else ignore_axis
                cshape = [np.insert(cshape, insert_axis,
                                    data_shapes[idx][ignore_axis])
                          for idx, s in enumerate(signals)]
            broad_signals = [broadcast_cshape(s, tuple(cshape[idx]))
                             for idx, s in enumerate(signals)]
            return broad_signals
        else:
            cshape = np.broadcast_shapes(*[s.cshape for s in signals])

    return [broadcast_cshape(s, cshape) for s in signals]


def broadcast_cdim(signal, cdim):
    """
    Broadcast a signal to a certain cdim.

    The channel dimension (cdim) is the length of the
    :py:mod:`cshape <pyfar._concepts.audio_classes>` of the signal. The signal
    is broadcasted to `cdim` by prepending ``cdim - len(signal.cshape)``
    dimensions.

    Parameters
    ----------
    signal : Signal, TimeData, FrequencyData
        The signal to be broadcasted.
    cdim : int
        The cdim to which `signal` is broadcasted.

    Returns
    -------
    signal : Signal, TimeData, FrequencyData
        The broadcasted copy input signal
    """

    if not isinstance(signal, (pf.Signal, pf.TimeData, pf.FrequencyData)):
        raise TypeError("Input data must be a pyfar audio object")
    if len(signal.cshape) > cdim:
        raise ValueError(
            "Can not broadcast: Current channel dimensions exceeds cdim")

    signal = signal.copy()
    while len(signal.cshape) < cdim:
        signal._data = signal._data[None, ...]
    return signal


def broadcast_cdims(signals, cdim=None):
    """
    Broadcast multiple signals to a common cdim.

    The channel dimension (cdim) is the length of the
    :py:mod:`cshape <pyfar._concepts.audio_classes>` of the signal. The signals
    are broadcasted to `cdim` by prepending ``cdim - len(signal.cshape)``
    dimensions.

    Parameters
    ----------
    signals : tuple of Signal, TimeData, FrequencyData
        The signals to be broadcasted in a tuple.
    cdim : int, optional
        The cdim to which `signal` is broadcasted. If `cdim` is ``None`` the
        signals are broadcasted to the largest `cdim`. The default is ``None``.

    Returns
    -------
    signals : tuple of Signal, TimeData, FrequencyData
        The broadcasted copies of the input signals in a tuple.
    """

    for signal in signals:
        if not isinstance(signal, (pf.Signal, pf.TimeData, pf.FrequencyData)):
            raise TypeError("All input data must be pyfar audio objects")
    if cdim is None:
        cdim = np.max([len(s.cshape) for s in signals])
    return [broadcast_cdim(s, cdim) for s in signals]

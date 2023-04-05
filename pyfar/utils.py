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


def broadcast_cshapes(signals, cshape=None):
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


def concatenate(signals, caxis=0, broadcasting=False, comment=""):
    """
    Merge multiple Signal objects along a given axis. The signals are copied,
    if needed broadcasted in largest dimenson and to a common cshape and a new
    object is returned.

    The :py:mod:`cshape <pyfar._concepts.audio_classes>` of the signals are
    broadcasted following the `numpy broadcasting rules
    <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_

    Parameters
    ----------
    signals : tuple of Signal
        The signals to concatenate. Must either have the same cshape or be
        broadcastable to the same cshape, except in the dimension corresponding
        to axis (the first, by default). If this is the case, set
        ``broadcasting=True``.
    axis : int
        The axis along which the signals are concatenated. The default is 0.
    broadcasting: bool
        If this is ``True``, the signals will be broadcasted into largest
        dimension and into a common cshape, except of the axis along which the
        signals are concatenated. The default is ``False``.
    comment: string
        A comment related to the merged `data`. The default is ``""``, which
        initializes an empty string. Comments of the input signals will also
        be returned in the new Signal object. These comments are marked with
        their corresponding signal position in the input tuple.
    Returns
    -------
    merged : Signal
        The merged signal object.
    """
    # check input
    for signal in signals:
        if not isinstance(signal, pf.Signal):
            raise TypeError("All input data must be of type pf.Signal")
    if not isinstance(comment, str):
        raise TypeError("'comment' needs to be a string.")
    if not isinstance(broadcasting, bool):
        raise TypeError("'broadcasting' needs to be False or True.")
    # check matching meta data of input signals.
    [signals[0]._assert_matching_meta_data(s) for s in signals]
    # broadcast signals into largest dimension and common cshapes
    if broadcasting is True:
        # broadcast signals into common cshape
        # signals = pf.utils.broadcast_cshapes(signals, ignore_axis=caxis)
        cshapes = [s.cshape for s in signals]
        max_cdim = np.max([len(sh) for sh in cshapes])
        for i, sh in enumerate(cshapes):
            for _ in range(max_cdim-len(sh)):
                cshapes[i] = (1,) + cshapes[i]
        # Finds broadcast cshape without the caxis to ignore
        cshape = np.broadcast_shapes(*np.delete(cshapes, caxis,
                                     axis=-1))
        broad_signals = []
        for i, s in enumerate(signals):
            # Appends the axis to ignore back into cshape to broadcast to.
            if caxis in (-1, len(cshape)):
                # Use append if ignore_axis is defined for last dimension
                cs = np.append(cshape, cshapes[i][caxis])
            else:
                # Use insert if ignore_axis is not defined for last dim
                axis = caxis+1 if caxis < 0 else caxis
                cs = np.insert(cshape, axis, cshapes[i][caxis])
            broad_signals.append(broadcast_cshape(s, tuple(cs)))
        signals = broad_signals
    # merge the signals along axis
    axis = caxis-1 if caxis < 0 else caxis
    data = np.concatenate([s._data for s in signals], axis=axis)
    # append comments in signals to comment to return
    if comment != "":
        comment = comment + '\n'
    comment = comment + f'Signals concatenated in caxis={caxis}.\n'
    sig_channel = 0
    for s in signals:
        if s.comment != "":
            comment += f"Channel {sig_channel+1}-"\
                       f"{sig_channel + s.cshape[caxis]}: "\
                       + s.comment + '\n'
        sig_channel += s.cshape[caxis]
    # return merged Signal
    return pf.Signal(data, signals[0].sampling_rate,
                     n_samples=signals[0].n_samples,
                     domain=signals[0].domain,
                     fft_norm=signals[0].fft_norm,
                     comment=comment)

"""
The utilities contain functions that are helpful when working with multiple
pyfar audio objects. The pyfar gallery gives background information to
:ref:`work with audio objects </gallery/interactive/pyfar_audio_objects.ipynb>`
including an introduction to the channel shape (`cshape`), channel axis
(`caxis`), and channel dimension (`cdim`).
"""
import pyfar as pf
import numpy as np


def broadcast_cshape(signal, cshape):
    """
    Broadcast a signal to a certain cshape.

    The channel shape (`cshape`) gives the shape of the audio data excluding
    the last dimension, which is ``n_samples`` for time domain objects and
    ``n_bins`` for frequency domain objects. The broadcasting follows the
    :doc:`numpy broadcasting rules <numpy:user/basics.broadcasting>`.

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

    The channel shape (`cshape`) gives the shape of the audio data excluding
    the last dimension, which is ``n_samples`` for time domain objects and
    ``n_bins`` for frequency domain objects. The broadcasting follows the
    :doc:`numpy broadcasting rules <numpy:user/basics.broadcasting>`.

    Parameters
    ----------
    signals : tuple of Signal, TimeData, FrequencyData
        The signals to be broadcasted in a tuple.
    cshape : tuple, optional
        The cshape to which the signals are broadcasted. If `cshape` is
        ``None`` it is determined from the cshapes of the input signals using
        :py:func:`numpy:numpy.broadcast_shapes`. The default is ``None``.

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

    The channel dimension (`cdim`) gives the dimension of the audio data
    excluding the last dimension, which is ``n_samples`` for time domain
    objects and ``n_bins`` for frequency domain objects. The signal is
    broadcasted to `cdim` by prepending ``cdim - len(signal.cshape)``
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

    The channel dimension (`cdim`) gives the dimension of the audio data
    excluding the last dimension, which is ``n_samples`` for time domain
    objects and ``n_bins`` for frequency domain objects. The signals are
    broadcasted to `cdim` by  prepending ``cdim - len(signal.cshape)``
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


def concatenate_channels(signals, caxis=0, broadcasting=False):
    """
    Merge multiple Signal, Timedata or Frequencydata objects along a given
    caxis.

    Parameters
    ----------
    signals : tuple of Signal, TimeData or FrequencyData
        The signals to concatenate. All signals must be of the same object type
        and either have the same cshape or be broadcastable to the same cshape,
        except in the dimension corresponding to caxis (the first, by default).
        If this is the case, set ``broadcasting=True``.
    caxis : int
        The channel axis (`caxis`) along which the signals are concatenated.
        The channel axis gives the axe of the audio data excluding the last
        dimension, which is ``n_samples`` for time domain objects and
        ``n_bins`` for frequency domain objects. The default is ``0``.
    broadcasting: bool
        If this is ``True``, the signals will be broadcasted to common
        cshape, except for the caxis along which the signals are
        concatenated.
        The caxis of the signals are broadcasted following the
        :doc:`numpy broadcasting rules <numpy:user/basics.broadcasting>`
        The default is ``False``.
    Returns
    -------
    merged : Signal, TimeData, FrequencyData
        The merged signal object.
    """
    # check input
    for signal in signals:
        if not isinstance(signal, (pf.Signal, pf.TimeData, pf.FrequencyData)):
            raise TypeError("All input data must be of type pf.Signal, "
                            "pf.TimeData or pf.FrequencyData.")
    if not isinstance(broadcasting, bool):
        raise TypeError("'broadcasting' needs to be False or True.")
    # check matching meta data of input signals.
    [signals[0]._assert_matching_meta_data(s) for s in signals]

    # broadcast signals into largest dimension and common cshapes
    if broadcasting is True:
        # broadcast signals into common cshape
        cshapes = [s.cshape for s in signals]
        max_cdim = np.max([len(sh) for sh in cshapes])
        for i, sh in enumerate(cshapes):
            for _ in range(max_cdim-len(sh)):
                cshapes[i] = (1,) + cshapes[i]
        # Finds broadcast cshape without the caxis to ignore
        cshape_bc = np.broadcast_shapes(*np.delete(cshapes, caxis,
                                        axis=-1))
        broad_signals = []
        for signal, cshape in zip(signals, cshapes):
            # Appends the caxis to ignore back into cshape to broadcast to.
            if caxis in (-1, len(cshape_bc)):
                # Use append if caxis is defined for last dimension
                cs = np.append(cshape_bc, cshape[caxis])
            else:
                # Use insert if caxis is not defined for last dim
                axis = caxis+1 if caxis < 0 else caxis
                cs = np.insert(cshape_bc, axis, cshape[caxis])
            broad_signals.append(broadcast_cshape(signal, tuple(cs)))
        signals = broad_signals
    # merge the signals along axis
    axis = caxis-1 if caxis < 0 else caxis
    # distinct by type using s._data does not work
    # - shapes of data would not match in mixed domain concatenation
    # - fft_norm may cause wrong time domain amplitudes in frequency domain
    #   concatenation
    if type(signal) is pf.FrequencyData:
        data = np.concatenate([s.freq for s in signals], axis=axis)
    else:
        data = np.concatenate([s.time for s in signals], axis=axis)
    # return merged Signal
    if isinstance(signals[0], pf.Signal):
        return pf.Signal(data, signals[0].sampling_rate,
                         n_samples=signals[0].n_samples,
                         domain=signals[0].domain,
                         fft_norm=signals[0].fft_norm)
    elif isinstance(signals[0], pf.TimeData):
        return pf.TimeData(data, signals[0].times)
    else:
        return pf.FrequencyData(data, signals[0].frequencies)

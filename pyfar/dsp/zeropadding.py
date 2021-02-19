from pyfar import Signal
import numpy as np

def zeropadding(signal, count, pos = 'front'):
    """Returns a time signal of type Signal padded with zeros
    
    Parameters
    ----------
    signal: Signal
        The signal that should be padded
    count: int
        the amount of zeros
    pos: string
        the position 'front' or 'back', standard 'front'

    Returns
    -------
    padded_signal : Signal
        The input signal with a zero padded time domain.
    
    """
    if(type(signal) != Signal):
        assert("signal must be of type Signal")
    
    signal_flat = signal.flatten()

    new_length = signal.n_samples + count
    padded_signal = Signal(np.zeros([signal_flat.cshape[0], new_length]),\
                            signal.sampling_rate)

    for channel in range(0, signal_flat.cshape[0]):
        pos.lower()
        if pos == 'front':
            padded_signal.time[channel][count:] = signal_flat.time[channel].copy()
        
        elif pos == 'back':
            padded_signal.time[channel][:-count] = signal_flat.time[channel].copy()

        else:
            assert("position must be 'front' or 'back'")

    return padded_signal.reshape(signal.cshape)
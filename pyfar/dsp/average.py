def average(signal, average_mode='time', phase_copy=None,
            weights=None):
    """
    Can be used to average impulse responses in different ways.

    Parameters
    ----------
    signal: Signal
        Input signal of the signal class
    average_mode: string
        'time' - averages in time domain
        'complex' - averages the complex spectra
        'abs' - averages the magnitude spectra
        'power' - averages the power spectra
        'log_magnitude' - averages the log magnitude spectra
        The default is 'time'
    phase_copy: vector
        indicates signal channel from which phase is to be coppied
        to the averaged signal
        None - ignores the phase. Results in zero phase
        The default is None
    weights: array
        array that gives weights for averaging the data
        The default is None
    Returns
    --------
    averaged_signal: Signal
        averaged impulse responses
    """

    # check input
    if not isinstance(signal, Signal):
        raise TypeError('Input data has to be of type: Signal')

    # check if 'weights' is right size

    # average the data

    if average_mode == 'time':
        if average_weights is None:
            data = np.mean(signal.time.copy(), axis=-1, keepdims=True)
        else:
            data = np.sum(signal.time.copy() * weights, axis=-1,
                          keepdims=True)

    elif average_mode == 'complex':
        if average_weights is None:
            data = np.mean(signal.freq.copy(), axis=-1, keepdims=True)
        else:
            data = np.sum(signal.freq.copy() * weights, axis=-1,
                          keepdims=True)

        # do I need to ifft here, or just put straight back into signal

    elif average_mode == 'abs':
        data = np.abs(signal.freq.copy())
        if average_weights is None:
            data_mag = np.mean(data, axis=-1, keepdims=True)
        else:
            data_mag = np.sum(data * weights, axis=-1, keepdims=True)

    elif average_mode == 'power':
        data = (np.abs(signal.freq.copy()))**2
        if average_weights is None:
            data_mag = np.sqrt(np.mean(data, axis=-1, keepdims=True))
        else:
            data_mag = np.sqrt(np.sum(data * weights, axis=-1, keepdims=True))

    elif average_mode == 'log_magnitude':
        data = 20 * np.log10(signal.freq.copy())
        if average_weights is None:
            data_mag = np.mean(data, axis=-1, keepdims=True)
            data_mag = 10**(averaged/20)
        else:
            data_mag = np.sqrt(np.sum(data * weights, axis=-1, keepdims=True))
            data_mag = 10**(averaged/20)

    else:
        raise ValueError(
            "average_mode must be 'time', 'complex', 'abs', 'power' or
            'log_magnitude'"
            )

    # phase handling
    if phase_copy is None:
        pass

    # unsure of how to access and edit phase data in signal class
    else:
        # copy phase from particular channel from phase_copy
        data_ang =
        # insert into 'averaged'
       data = data_mag * exp(1j * data_ang)
        # ifft 
        data = 

    
    # replace input with normalized_input
    averaged_signal = signal.copy()
    averaged_signal.time = data

    return averaged_signal

import numpy as np

def normalize(input_signal, norm_type = 'time', operation = 'max', channelwise = 'max',
              value = 'unnasigned', f = np.arange(0,22100),  fs = 44100):
    """
    Parameters
    ----------
    input_signal:
        signal of the signal class
        
    norm_type:
        'time' - normalizes time signal to 'value' 
        'abs' - normalizes magnitude spectrum to 'value'
        'dB' - normalizes db magnitude spectrum to 'value'
        (default = 'time')

    operation:
        'max' - finds absolute maximum of data to normalize 
        'mean' - find the mean of data to normalize
        'rms' - fins the rms of data to normalize
        (default = 'max')
    
    channelwise:
        'each' - normalize each channel seperately 
        'max' - normalize to max or operation across all channels
        'min' - normalize to min or operation across all channels
        'mean' - normalize to mean of operation across all channels
        (default = 'max')
    
    value: 
        normalizes to 'value' which can be a scalor or a vector 
        with a number of elements equal to channels. Default is 0 
        for type = 'db' and 1 otherwise
    f:
        two element vector specifiying upper and lower frequenzy bounds
        for normalization or scalar specifying the centre frequency for 
        normalization (see 'frac', default is [0,fs/2])
    fs:
        sampling frequency in Hz (default = 44100)
        
    Returns
    --------
    signal_normalized - normalized input signal
    """
    
    # set defaults
    
    if value == 'unnasigned':
        if norm_type == 'dB':
            value = 0
        else:
            value = 1
    
    # copy input
    
    input_normalized = input_signal.time.copy()

    # flatten
    
    #input_normalized =  input_normalized.flatten() # Flattening / Reshaping to a 2d array to work with possible?

    # transform data to the desired domain
    
    if norm_type == 'time':
         input_normalized = input_normalized
    
    elif norm_type == 'abs':
        input_normalized = np.abs(np.fft.fft(input_normalized,input_normalized.ndim - 1)) 
    
    elif norm_type == 'dB':
        input_normalized = np.fft.fft(input_normalized,input_normalized.ndim - 1)
        
    else:
        raise ValueError(("norm _type must be 'time', 'abs' or 'dB'"))
      
    # get bounds for normalization
    
    if norm_type == 'time':
        lim = np.array([0, input_normalized.shape[1]])
    
    else: # if domain is abs or db
        
        if len(f) == 1:
            #frequency spacing
            df = fs/(input_normalized.shape[1])
            
            # get frequency limits
            f = f/2^(1/2)
            f = np.array([f, f*2])
            
            # get corresponding indicees
            lim = np.round(f/df) + 1
            
        else: 
            # frequency spacing
            df = fs/(input_normalized.shape[1])
            
            # get corresponding indices
            lim = np.round(f/df) + 1
              
    # get values for normalization
    
    if operation == 'max':
        if norm_type == 'time':
            values = np.max(np.abs(input_normalized[:,lim[0]:lim[1]]),input_normalized.ndim - 1)
        else: 
            values = np.max(input_normalized[:,lim[0]:lim[1]],input_normalized.ndim - 1)
    
    elif operation == 'mean':
        values = np.mean(input_normalized[:,lim[0]:lim[1]],input_normalized.ndim - 1)
    
    elif operation == 'rms':
        values = np.sqrt(np.mean(((input_normalized[:,lim[0]:lim[1]])**2),1),input_normalized.ndim - 1)
    
    else: 
        raise ValueError(("'operation' must be 'abs', 'mean' or 'rms'"))
        
    
    if norm_type == 'db':
        valuesLin = 10^(values/20)
        valueLin = 10^(value/20)
    
    else:
        valuesLin = values
        valueLin = value
        
    # apply normalization
     
    if channelwise == 'each':
        input_normalized = np.divide(input_normalized.T,valuesLin).T

    elif channelwise == 'max':
        input_normalized = input_normalized / np.max(valuesLin)
        
    elif channelwise == 'min':
        input_normalized = input_normalized / np.min(valuesLin)
    
    elif channelwise == 'mean':
        input_normalized = input_normalized / np.mean(valuesLin)
    else: 
        raise ValueError(("channelwise must be 'each', 'max', 'min' or  'mean'"))
    
    # normalize to value
                                                       
    input_normalized = input_normalized * valueLin
    
    # reshape
    
    #input_normalized = input_normalized.reshape(input_signal.cshape)
    
    return input_normalized
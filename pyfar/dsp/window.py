import numpy as np
from scipy import special as sp


# different window types; for even window lengths  window maximum not 1, especially for short windows!! set maximum to 1 by hand?
def rect(length):
    """returns a rectangular window of given length.

    Parameters
    ----------
    length : integer
        window length in samples

    Returns
    -------
    out : np.array()
        window coefficients
        
    Reference
    ---------
       [1] Oppenheim, A. V., & Schafer, R. W. (2010). Discrete-time Signal Processing. Pearson.
    """
    
    if length > 0:
        return np.ones(length)
    else:
        raise ValueError('window length has to be >=0 !')
    

def hann(length):
    """returns a hann window of given length.

    Parameters
    ----------
    length : integer
        window length in samples

    Returns
    -------
    out : np.array()
        window coefficients
        
    Reference
    ---------
       [1] Oppenheim, A. V., & Schafer, R. W. (2010). Discrete-time Signal Processing. Pearson.
    """
    
    if length > 0:
        n = np.arange(0,length)
        return 0.5-0.5*np.cos(2*np.pi*n/(length-1))
    else:
        raise ValueError('window length has to be >=0 !')
  
    
def hamming(length):
    """returns a hamming window of given length.

    Parameters
    ----------
    length : integer
        window length in samples

    Returns
    -------
    out : np.array()
        window coefficients
        
    Reference
    ---------
      [1] Oppenheim, A. V., & Schafer, R. W. (2010). Discrete-time Signal Processing. Pearson.
    """
    
    if length > 0:
        n = np.arange(0,length)
        return 0.54-0.46*np.cos(2*np.pi*n/(length-1))
    else:
        raise ValueError('window length has to be >=0 !')


def blackman(length):
    """returns a blackman window of given length.

    Parameters
    ----------
    length : integer
        window length in samples

    Returns
    -------
    out : np.array()
        window coefficients
        
    Reference
    ---------
       [1] Oppenheim, A. V., & Schafer, R. W. (2010). Discrete-time Signal Processing. Pearson.
    """
    
    if length > 0:
        n = np.arange(0,length)
        return 0.42-0.5*np.cos(2*np.pi*n/(length-1)) + 0.08*np.cos(4*np.pi*n/(length-1))
    else:
        raise ValueError('window length has to be >=0 !')


def bartlett(length):
    """returns a bartlett window of given length.

    Parameters
    ----------
    length : integer
        window length in samples

    Returns
    -------
    out : np.array()
        window coefficients
        
    Reference
    ---------
       [1] Oppenheim, A. V., & Schafer, R. W. (2010). Discrete-time Signal Processing. Pearson.
    """
    
    if length > 0:
        n = np.arange(0,length)
        firstHalf = 2*n[0:length//2]/(length-1)
        secondHalf = 2-2*n[length//2:]/(length-1)
        return np.concatenate((firstHalf, secondHalf))
    else:
        raise ValueError('window length has to be >=0 !')     


def kaiser(length,A= 90.36969147005445):
    """returns a kaiser window of given length.

    Parameters
    ----------
    length : integer
        window length in samples
    A:
        approximate peak approximation error in dB ([1] formula 7.74)
        default value equals Kaiser beta = 9 window

    Returns
    -------
    out : np.array()
        window coefficients
        
    Reference
    ---------
       [1] Oppenheim, A. V., & Schafer, R. W. (2010). Discrete-time Signal Processing. Pearson.
    """
    
    if length >= 0:
        n = np.arange(0,length)
        # Kaiser window shape parameter: beta ([1] formula (7.75))
        if A > 50:
            beta = 0.1102*(A-8.7);
        elif A >= 21:
            beta = .5842*(A-21)**0.4 + .07886*(A-21);
        else:
            beta = 0

        alpha = (length-1)/2
        # [1] formula (7.72)
        return sp.iv(0,(beta*np.sqrt(1-((n-alpha)/alpha)**2))) / sp.iv(0,beta)
        
    else:
        raise ValueError('window length has to be >=0 !')
        



def kaiserBessel(length):
    """returns a kaiserBessel window of given length.

    Parameters
    ----------
    length : integer
        window length in samples

    Returns
    -------
    out : np.array()
        window coefficients
        
    Reference
    ---------
       [1] Oppenheim, A. V., & Schafer, R. W. (2010). Discrete-time Signal Processing. Pearson.
       [2] Meyer, M. (2017). Signalverarbeitung. Springer Publishing.
    """
    
    if length >= 0:
        n = np.arange(0,length)
        # [2] table 5.2 p.193
        return 0.4021 - 0.4986 * np.cos(2*np.pi*n/(length-1)) + 0.0981*np.cos(4*np.pi*n/(length-1)) - 0.0012 * np.cos(6*np.pi*n/(length-1))
    else:
        raise ValueError('window length has to be >=0 !')



def flattop(length):
    """returns a flattop window of given length.

    Parameters
    ----------
    length : integer
        window length in samples

    Returns
    -------
    out : np.array()
        window coefficients
        
    Reference
    ---------
       [1] Oppenheim, A. V., & Schafer, R. W. (2010). Discrete-time Signal Processing. Pearson.
       [2] Meyer, M. (2017). Signalverarbeitung. Springer Publishing.
    """
    
    if length >= 0:
        n = np.arange(0,length)
        # [2] table 5.2 p.193
        return 0.2155 - 0.4159 * np.cos(2*np.pi*n/(length-1)) + 0.278*np.cos(4*np.pi*n/(length-1)) - 0.0836 * np.cos(6*np.pi*n/(length-1)) + 0.007 * np.cos(8*np.pi*n/(length-1))
    else:
        raise ValueError('window length has to be >=0 !')
        

def dolphChebychev(length,sideLobeLevel = 60):
    """returns a dolphChebychev window of given length.

    Parameters
    ----------
    length : integer
        window length in samples

    Returns
    -------
    out : np.array()
        window coefficients
        
    Reference
    ---------
       [1] Oppenheim, A. V., & Schafer, R. W. (2010). Discrete-time Signal Processing. Pearson.
       [2] Meyer, M. (2017). Signalverarbeitung. Springer Publishing.
       [3] https://ccrma.stanford.edu/~jos/sasp/Dolph_Chebyshev_Window.html
    """
    
    if length >= 0:
        n = np.arange(0,length)
        # # [3]
        # alpha = sideLobeLevel/20
        # beta = np.cosh(1/length * np.arccosh(10**alpha))
        # W = np.cos(length * np.arccos(beta*np.cos(np.pi*n/length))) / np.cosh(length*np.arccosh(beta))# why do I always get NAN at the beginning and end?
        # # W[np.isnan(W)] = 0
        
        if length%2==0:# unsymmetric length, matlab makes it symmteric even with unsymmetric length...
            M = length
            gamma = sideLobeLevel/20
            alpha = np.cosh(np.arccosh(10**gamma)/M)
            m = np.arange(M)
            A = np.abs(alpha * np.cos(np.pi*m/M))
            
            W = np.zeros(np.size(A))
            for i_A in range(np.size(A)):
                if A[i_A] > 1:
                    W[i_A] = (-1)**m[i_A] * np.cosh(M* np.arccosh(A[i_A]))
                else:
                    W[i_A] = (-1)**m[i_A] * np.cos(M* np.arccos(A[i_A]))
                    
            w = np.real(np.fft.ifft(W))
            w[0] = w[0]/2
            # w = np.append(w, w[0])
            w = w/np.max(w)
        else: # symmetric length
            M = (length-1)
            gamma = sideLobeLevel/20
            alpha = np.cosh(np.arccosh(10**gamma)/M)
            m = np.arange(M)
            
            A = np.abs(alpha * np.cos(np.pi*m/M))
            
            W = np.zeros(np.size(A))
            for i_A in range(np.size(A)):
                if A[i_A] > 1:
                    W[i_A] = (-1)**m[i_A] * np.cosh(M* np.arccosh(A[i_A]))
                else:
                    W[i_A] = (-1)**m[i_A] * np.cos(M* np.arccos(A[i_A]))
                    
            w = np.real(np.fft.ifft(W))
            w[0] = w[0]/2
            w = np.append(w, w[0])
            w = w/np.max(w)
        return w
    else:
        raise ValueError('window length has to be >=0 !')
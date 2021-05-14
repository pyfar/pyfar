import pyfar
from pyfar import Signal
import numpy as np
import pyfar.dsp.Zeropadding

def test_zeropadding():
    numZeros = 100
    testSignal = Signal(np.ones(15, 3, 1024), 44100)
    
    padded_front = pyfar.dsp.Zeropadding.zeropadding(testSignal, numZeros)
    assert testSignal.cshape == padded_front.cshape 
    assert testSignal.n_samples + numZeros == padded_front.n_samples
    for i in range(padded_front.cshape[0]):
        for j in range(padded_front[0].cshape[1]):
            for sample in range(padded_front.n_samples):
                if sample < numZeros:
                    assert padded_front[i][j][sample] == 0
                else:
                    assert padded_front[i][j][sample] == 1

    padded_back = pyfar.dsp.Zeropadding.zeropadding(testSignal,\
                                                    numZeros, 'back')
    assert testSignal.cshape == padded_back.cshape
    assert testSignal.n_samples + numZeros == padded_back.n_samples
    for i in range(padded_back.cshape[0]):
        for j in range(padded_back.cshape[1]):
            for sample in range(padded_back.n_samples):
                if sample < testSignal.n_samples:
                    assert padded_back[i][j][sample] == 1
                else:
                    assert padded_back[i][j][sample] == 0

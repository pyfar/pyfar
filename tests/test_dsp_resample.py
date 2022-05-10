import pyfar
# import pytest
# import numpy as np


def test_resample():
    # Will rewrite these test, this i just for a first push. 
    signal = pyfar.Signal([0.1, 0.2, 0.3], 44100)
    resampled_sig = pyfar.dsp.resample(signal, 44100*2)
    assert resampled_sig.n_samples == 6

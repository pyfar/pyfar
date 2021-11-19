import numpy as np
from pyfar import Signal
from pyfar.dsp import convolve_overlap_add


def test_overlap_add(impulse):
    ir = [1, 0.5, 0.25, 0, 0, 0, 0, 0]
    ir_sig = Signal(ir, impulse.sampling_rate)

    res = convolve_overlap_add(impulse, ir_sig, mode='full')

    desired = np.zeros((1, impulse.n_samples + len(ir) - 1))
    desired[:, :len(ir)] = ir
    np.testing.assert_allclose(res.time, desired, atol=1e-10)

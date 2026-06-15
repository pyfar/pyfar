import numpy as np
import numpy.testing as npt
import pyfar as pf
from pyfar.classes.filter import StateSpaceModel


def _fir_state_space(b):
    """Construct a state-space realization (A,B,C,D) of an FIR filter b."""
    b = np.asarray(b).ravel()
    L = b.size
    if L == 1:
        # zero-order FIR (pure feedthrough)
        A = np.zeros((0, 0))
        B = np.zeros((0, 1))
        C = np.zeros((1, 0))
        D = np.array([[b[0]]])
        return A, B, C, D
    order = L - 1
    A = np.zeros((order, order))
    # ones on the first subdiagonal to shift previous inputs
    for i in range(1, order):
        A[i, i-1] = 1.0
    B = np.zeros((order, 1))
    B[0, 0] = 1.0
    C = b[1:].reshape(1, -1)
    D = np.array([[b[0]]])
    return A, B, C, D


def test_state_space_impulse_response_matches_fir():
    b = np.array([0.2, 0.3, 0.5])
    A, B, C, D = _fir_state_space(b)
    ss = StateSpaceModel(A, B, C, D, sampling_rate=48000)

    # impulse response should match b (padded to requested length)
    h = ss.impulse_response(len(b))
    # h.time has shape (n_inputs, n_outputs, n_samples)
    npt.assert_allclose(h.time[0, 0, :len(b)], b)


def test_state_space_process_matches_convolution():
    # construct simple FIR and its state-space realization
    b = np.array([0.2, 0.3, 0.5])
    A, B, C, D = _fir_state_space(b)
    ss = StateSpaceModel(A, B, C, D, sampling_rate=44100)

    rng = np.random.RandomState(0)
    x = rng.randn(256)
    sig = pf.Signal(x, 44100)

    y_ss = ss.process(sig).time.squeeze()

    # expected convolution (causal FIR)
    expected = np.convolve(x, b)[: x.size]

    npt.assert_allclose(y_ss, expected, atol=1e-12)

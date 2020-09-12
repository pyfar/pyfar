import numpy as np
from haiopy import Signal

a = np.arange(6).reshape(2, 3)
for x in np.nditer(a):
    print(x, end=' ')


for e in a:
    print(e)


sig = Signal(np.random.randn(2, 3, 2**10), 1e3, signal_type='power')
sig = Signal(np.random.randn(3, 2**10), 1e3, signal_type='power')
sig = Signal(np.random.randn(2**10), 1e3, signal_type='power')

# for s in sig:
#     print(s.shape)


class SignalIterator(object):
    def __init__(self, array_iterator, signal):
        self._array_iterator = array_iterator
        self._signal = signal
        self._iterated_sig = Signal(
            None,
            sampling_rate=signal.sampling_rate,
            n_samples=signal.n_samples,
            domain=signal.domain,
            signal_type=signal.signal_type,
            dtype=signal.dtype)

    def __next__(self):
        if self._signal.domain == self._iterated_sig.domain:
            data = self._array_iterator.__next__()
            self._iterated_sig._data = data
        else:
            raise RuntimeError("domain changes during iterations break stuff!")

        return self._iterated_sig


sig.__iter__ = SignalIterator(sig._data.__iter__, sig)

for s in sig:
    s.time = np.ones(1024)


# from numpy.lib.arrayterator import Arrayterator
# arrayterator_sig = Arrayterator(sig, buf_size=3)


# for a in arrayterator_sig:
#     print(a)


# arrayterator_ndarray = Arrayterator(sig.time)

# for a in arrayterator_ndarray:
#     print(a)

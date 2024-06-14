import pyfar as pf
import matplotlib.pyplot as plt
import numpy as np
test = pf.Signal(np.array([1, 2, 3, 5, 6, 13, 3, 3, 2, 2]),
                 sampling_rate=48000, is_complex=True)

pf.plot.time(test, show_real_imag_abs='imag')

pf.plot.freq(test, side='right')

plt.show()

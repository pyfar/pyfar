# %%
import matplotlib.pyplot as plt
import numpy as np
import pyfar
import pyfar.dsp.dsp as dsp
from scipy.signal import chirp


sr = 44100
times = np.arange(2**16)/sr

sweep = pyfar.Signal(
    chirp(times, 100, times[-1], 20e3, method='logarithmic'), sr)
inv = dsp.regularized_spectrum_inversion(sweep, (100, 20e3))
# %%
plt.semilogx(inv.frequencies, 20*np.log10(np.abs(inv.freq.T)))
# %%

pyfar.plot.freq(inv)

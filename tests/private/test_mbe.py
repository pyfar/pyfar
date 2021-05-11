# %%
%matplotlib qt
# %%
from scipy.signal.ltisys import impulse
import pyfar

# %%

imp = pyfar.signals.impulse(2**9, amplitude=[[1, 1], [1, 1]])

# %%
imp.time

# %%
signal_shifted = pyfar.dsp.time_shift(imp, [[20, 30], [25, 35]])
signal_shifted = pyfar.dsp.time_shift(imp, 10)
pyfar.plot.time(signal_shifted.flatten(), unit='samples')

# %%

# %%
signal_shifted = pyfar.dsp.phase_shift(imp, 0.7)
pyfar.plot.time(signal_shifted.flatten(), unit='samples')

# %%

# %%
imp = pyfar.signals.impulse(2**9)

# %%
import numpy as np
import scipy.signal as sps
sps.minimum_phase(np.squeeze(imp.time[..., :-1]), n_fft=2048).shape

# %%
imp = pyfar.signals.impulse(512, delay=10)
imp.freq *= np.exp(-1j*np.pi/2)
imp_minphase = pyfar.dsp.minimum_phase(imp, method='homomorphic')
pyfar.plot.time(imp_minphase)
# %%



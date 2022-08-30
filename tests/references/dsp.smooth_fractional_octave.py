# %% Write smoothed signal for testing.
# The result was inspected manually before writing the data.
import numpy as np
import matplotlib.pyplot as plt
import pyfar as pf
pf.plot.use()

# generate signal from two bell filters
signal = pf.signals.impulse(441)
signal = pf.dsp.filter.bell(signal, 1e3, 12, 1, "III")
signal = pf.dsp.filter.bell(signal, 10e3, -60, 100, "III")
np.savetxt("dsp.smooth_fractional_octave_input.csv", signal.time)

signal = pf.Signal(np.loadtxt("dsp.smooth_fractional_octave_input.csv"),
                   44100)

# test different modes --------------------------------------------------------
plt.figure()
ax = pf.plot.time_freq(signal)

y, _ = pf.dsp.smooth_fractional_octave(signal, 1, mode="magnitude_zerophase")
pf.plot.time_freq(y, label="magnitude_zerophase")
np.savetxt("dsp.smooth_fractional_octave_magnitude_zerophase.csv", y.time)

y, _ = pf.dsp.smooth_fractional_octave(signal, 1, mode="magnitude_phase")
pf.plot.time_freq(y, label="magnitude phase")
np.savetxt("dsp.smooth_fractional_octave_magnitude_phase.csv", y.time)

y, _ = pf.dsp.smooth_fractional_octave(signal, 1, mode="magnitude")
pf.plot.time_freq(y, label="magnitude copy")
np.savetxt("dsp.smooth_fractional_octave_magnitude_copy_phase.csv", y.time)

y, _ = pf.dsp.smooth_fractional_octave(signal, 1, mode="complex")
pf.plot.time_freq(y, label="complex")
np.savetxt("dsp.smooth_fractional_octave_complex.csv", y.time)

ax[1].legend(loc=3)

# test different smothing widths ----------------------------------------------
plt.figure()
ax = pf.plot.time_freq(signal)

y, _ = pf.dsp.smooth_fractional_octave(signal, 1)
pf.plot.time_freq(y, label="1")
np.savetxt("dsp.smooth_fractional_octave_1.csv", y.time)

y, _ = pf.dsp.smooth_fractional_octave(signal, 5)
pf.plot.time_freq(y, label="0.2")
np.savetxt("dsp.smooth_fractional_octave_5.csv", y.time)

ax[1].legend(loc=3)

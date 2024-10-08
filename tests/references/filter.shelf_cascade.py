# Write cascaded shelf filters for testing.
# The filter shape was inspected manually before writing the data.
import numpy as np
import pyfar as pf

x = pf.signals.impulse(2**10)

# low shelf cascade
y, *_ = pf.dsp.filter.low_shelf_cascade(x, 4e3, "upper", -20, None, 4)
pf.plot.freq(y)
np.savetxt("filter.shelve_cascade_low.csv", y.time)

# high shelf cascade
y, *_ = pf.dsp.filter.high_shelf_cascade(x, 250, "lower", -20, None, 4)
pf.plot.freq(y)
np.savetxt("filter.shelve_cascade_high.csv", y.time)

# high shelf cascade (upper frequency exceeds Nyquist)
y, *_ = pf.dsp.filter.high_shelf_cascade(x, 22050/2, "lower", -20, None, 2)
pf.plot.freq(y)
np.savetxt("filter.shelve_cascade_high_exceed_nyquist.csv", y.time)

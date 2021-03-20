# Write linear sweep to csv for testing.
# The sweep was manually inspected.
import numpy as np
from pyfar.signals import linear_sweep

sweep = linear_sweep(2**10, [1e3, 20e3]).time
np.savetxt("signals.linear_sweep.csv", sweep)

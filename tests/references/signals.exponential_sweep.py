# Write exponential sweep to csv for testing.
# The sweep was manually inspected.
import numpy as np
from pyfar.signals import exponential_sweep

sweep = exponential_sweep(2**10, [1e3, 20e3]).time
np.savetxt("signals.exponential_sweep.csv", sweep)

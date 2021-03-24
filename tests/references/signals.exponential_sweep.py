# Write exponential sweep to csv for testing.
# The sweep was manually inspected.
# The time signal was inspected for smootheness and maximum amplitudes of +/-1.
# The spectrum was inspected for the ripple at the edges of the frequency range
# (typical for time domain sweep generation) and the 1/f slope.
import numpy as np
from pyfar.signals import exponential_sweep

sweep = exponential_sweep(2**10, [1e3, 20e3]).time
np.savetxt("signals.exponential_sweep.csv", sweep)

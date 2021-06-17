# Write linear sweep to csv for testing.
# The sweep was manually inspected.
# The time signal was inspected for smootheness and maximum amplitudes of +/-1.
# The spectrum was inspected for the ripple at the edges of the frequency range
# (typical for time domain sweep generation) and constant amplitude across
# frequency.
import numpy as np
from pyfar.signals import linear_sweep_time

sweep = linear_sweep_time(2**10, [1e3, 20e3]).time
np.savetxt("signals.linear_sweep_time.csv", sweep)

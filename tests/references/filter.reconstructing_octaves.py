# Write reconstructing fractional octave filters for testing.
# The filter shape was inspected manually before writing the data.
import numpy as np
import pyfar as pf

x = pf.signals.impulse(2**10)

# overlap=1, slope=0
y, _ = pf.dsp.filter.reconstructing_fractional_octave_bands(
    x, frequency_range=(8e3, 16e3), overlap=1, slope=0, n_samples=2**10)
np.savetxt("filter.reconstructing_octaves_1_0.csv", y.time)

# overlap=0, slope=0
y, _ = pf.dsp.filter.reconstructing_fractional_octave_bands(
    x, frequency_range=(8e3, 16e3), overlap=0, slope=0, n_samples=2**10)
np.savetxt("filter.reconstructing_octaves_0_0.csv", y.time)

# overlap=1, slope=3
y, _ = pf.dsp.filter.reconstructing_fractional_octave_bands(
    x, frequency_range=(8e3, 16e3), overlap=1, slope=3, n_samples=2**10)
np.savetxt("filter.reconstructing_octaves_1_3.csv", y.time)

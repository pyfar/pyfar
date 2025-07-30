# %% Generate test data for pyfar.constants.octave_band_tolerance
#    Data checked by inspection of numbers and plots upon generation.
import pyfar as pf
from pyfar.constants import octave_band_tolerance
import matplotlib.pyplot as plt
import numpy as np
pf.plot.use()


for exact_center_frequency in [1000, 1000 * 10**.3]:
    for bands in ['octave', 'third']:
        for tolerance_class in [1, 2]:

            tolerance = octave_band_tolerance(
                exact_center_frequency, bands, tolerance_class)

            # plot for visual inspection
            plt.fill_between(
                tolerance.frequencies, tolerance.freq[0], tolerance.freq[1],
                facecolor='g', alpha=.25, label='Class 1 Tolerance')
            ax = plt.gca()
            ax.set_xlim(np.min(tolerance.frequencies),
                        np.max(tolerance.frequencies))
            ax.set_ylim(np.min(tolerance.freq[1]), 2.5)
            ax.set_xscale('log')
            ax.set_title(f'f={exact_center_frequency} Hz, {bands=}, tolerance '
                         f'{tolerance_class}')

            # Save tolerance data to text file
            filename = (f"octave_band_tolerance_{int(exact_center_frequency)}"
                        f"Hz_{bands}_class{tolerance_class}.csv")

            data = np.vstack((tolerance.frequencies, tolerance.freq))
            np.savetxt(filename, data, fmt="%.2f", delimiter=', ')
            plt.show()

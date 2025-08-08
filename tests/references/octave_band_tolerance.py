# %% Generate test data for pyfar.constants.octave_band_tolerance
#    Data checked by inspection of numbers and plots upon generation.
import pyfar as pf
from pyfar.constants import octave_band_tolerance
import matplotlib.pyplot as plt
import numpy as np
pf.plot.use()


for exact_center_frequency in [1000, 1000 * 10**.3]:
    for num_fractions in [1, 3]:
        for tolerance_class in [1, 2]:

            tolerance, frequencies = octave_band_tolerance(
                exact_center_frequency, num_fractions, tolerance_class)

            # plot for visual inspection
            plt.fill_between(
                frequencies, tolerance[0], tolerance[1],
                facecolor='g', alpha=.25, label='Class 1 Tolerance')
            ax = plt.gca()
            ax.set_xlim(np.min(frequencies),
                        np.max(frequencies))
            ax.set_ylim(np.min(tolerance[1]), 2.5)
            ax.set_xscale('log')
            ax.set_title(
                f'f={exact_center_frequency} Hz, {num_fractions=}, tolerance '
                f'{tolerance_class}')

            # Save tolerance data to text file
            filename = (f"octave_band_tolerance_{int(exact_center_frequency)}"
                        f"Hz_num_fractions{num_fractions}_"
                        f"class{tolerance_class}.csv")

            data = np.vstack((frequencies, tolerance))
            np.savetxt(filename, data, fmt="%.2f", delimiter=', ')
            plt.show()

"""
Functions for testing of plots.

Intended to reduce code redundancy and assure reproducibility on different
operating systems
"""

import os
import matplotlib
import matplotlib.testing as mpt
import matplotlib.pyplot as plt
from matplotlib.testing.compare import compare_images


def create_figure(width=6, height=4.8, dpi=100):
    """
    Create figure with defined parameters for reproducible testing.

    Parameters
    ----------

    width : float
        The width in inch. The default is 6
    height : float
        The height in inch. The default is 4.8
    dpi : int
        The resolutoin in dots per inch. The default is 100

    Returns
    -------
    fig : Matplotlib Figure object
    """

    plt.close('all')
    matplotlib.use('Agg')
    mpt.set_reproducibility_for_testing()
    # force size/dpi for testing
    return plt.figure(1, (width, height), dpi)


def save_and_compare(create_baseline, baseline_path, test_path, filename,
                     file_type, compare_output):
    """
    Save current Figure as Image and compare images against baseline

    Parameters
    ----------
    create_baseline :  boolean
        If true, the current Figure is also used to create the baseline image
        against which everything is compared. This is usefull if the plot look
        changed.
    baseline_path : str
        Folder where the baseline images are saved
    test_path : str
        Folder where the images for testing against the baseline are saved
    filename : str
        The name with which the image is saved
    file_type : str
        The file type of the image file without the dot, e.g., ``'png'``
    compare_output : boolen
        If false the saved images are compared against each other and a diff
        image is saved under `test_oath` in case they differ using
        matplotlib.testing.compare.compare_images. If true, an error is also
        raised if the images differ.
    """
    # file names for saving
    baseline = os.path.join(baseline_path, filename + "." + file_type)
    output = os.path.join(test_path, filename + "." + file_type)

    # safe baseline and test image
    if create_baseline:
        plt.savefig(baseline)
    plt.savefig(output)

    # compare images
    comparison = compare_images(baseline, output, tol=10)
    if compare_output:
        assert comparison is None

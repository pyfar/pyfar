import pytest
import numpy as np
import os.path
import sofa
import scipy.io.wavfile as wavfile

from pyfar.orientations import Orientations
import pyfar.io

# import stub_utils


@pytest.fixture
def generate_orientations_file(tmpdir, orientations):
    """Create a far file in temporary folder that contains an orientations
    object
    """
    filename = os.path.join(tmpdir, 'test_orientations.far')
    pyfar.io.write(filename, orientations=orientations)
    return filename


# TODO: Merge the following with harmonizing_tests
@pytest.fixture
def views():
    return [[1, 0, 0], [2, 0, 0], [-1, 0, 0]]


@pytest.fixture
def ups():
    return [[0, 1, 0], [0, -2, 0], [0, 1, 0]]


@pytest.fixture
def positions():
    return [[0, 0.5, 0], [0, -0.5, 0], [1, 1, 1]]


@pytest.fixture
def orientations(views, ups):
    return Orientations.from_view_up(views, ups)
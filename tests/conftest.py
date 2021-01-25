from pyfar.coordinates import Coordinates
from pyfar.orientations import Orientations
from pyfar.Signal import Signal
import pyfar.io
import pytest
import numpy as np
import os.path
import sofa
import scipy.io.wavfile as wavfile


import stub_utils


@pytest.fixture
def generate_far_file(tmpdir, orientations, coordinates, signal):
    """Create a far file in temporary folder that contains an orientations
    object
    """
    filename = os.path.join(tmpdir, 'test_orientations.far')
    pyfar.io.write(
        filename,
        orientations=orientations,
        coordinates=coordinates,
        signal=signal)
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


@pytest.fixture
def coordinates():
    return Coordinates([0, 1], [2, 3], [4, 5])


# @pytest.fixture
# def signal(sine):
#     return Signal(sine, 44100, domain='time')

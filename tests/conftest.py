from pyfar.spatial.spatial import SphericalVoronoi
import pytest
import numpy as np
import os.path
import sofa
import scipy.io.wavfile as wavfile

from pyfar.orientations import Orientations
from pyfar.coordinates import Coordinates
from pyfar.signal import Signal
import pyfar.dsp.classes as fo
import pyfar.io

# import stub_utils


@pytest.fixture
def generate_far_file_orientations(tmpdir, orientations):
    """Create a far file in temporary folder that contains an Orientations
    object.
    """
    filename = os.path.join(tmpdir, 'test_orientations.far')
    pyfar.io.write(filename, orientations=orientations)
    return filename


@pytest.fixture
def generate_far_file_coordinates(tmpdir, coordinates):
    """Create a far file in temporary folder that contains an Coordinates
    object.
    """
    filename = os.path.join(tmpdir, 'test_coordinates.far')
    pyfar.io.write(filename, coordinates=coordinates)
    return filename


@pytest.fixture
def generate_far_file_signal(tmpdir, signal):
    """Create a far file in temporary folder that contains an Signal
    object.
    """
    filename = os.path.join(tmpdir, 'test_signal.far')
    pyfar.io.write(filename, signal=signal)
    return filename


@pytest.fixture
def generate_far_file_sphericalvoronoi(tmpdir, sphericalvoronoi):
    """Create a far file in temporary folder that contains an SphericalVoronoi
    object.
    """
    filename = os.path.join(tmpdir, 'test_sphericalvoronoi.far')
    pyfar.io.write(filename, sphericalvoronoi=sphericalvoronoi)
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


@pytest.fixture
def signal():
    # TODO: replace sine with fixture sine
    sine = np.sin(2 * np.pi * 440 * np.arange(0, 1, 1 / 44100))
    return Signal(sine, 44100, len(sine), domain='time')


@pytest.fixture
def sphericalvoronoi():
    dihedral = 2 * np.arcsin(np.cos(np.pi / 3) / np.sin(np.pi / 5))
    R = np.tan(np.pi / 3) * np.tan(dihedral / 2)
    rho = np.cos(np.pi / 5) / np.sin(np.pi / 10)

    theta1 = np.arccos(
        (np.cos(np.pi / 5) / np.sin(np.pi / 5)) /
        np.tan(np.pi / 3))

    a2 = 2 * np.arccos(rho / R)

    theta2 = theta1 + a2
    theta3 = np.pi - theta2
    theta4 = np.pi - theta1

    phi1 = 0
    phi2 = 2 * np.pi / 3
    phi3 = 4 * np.pi / 3

    theta = np.concatenate((
        np.tile(theta1, 3),
        np.tile(theta2, 3),
        np.tile(theta3, 3),
        np.tile(theta4, 3)))
    phi = np.tile(np.array(
            [phi1, phi2, phi3, phi1 + np.pi / 3,
             phi2 + np.pi / 3, phi3 + np.pi / 3]), 2)
    rad = np.ones(np.size(theta))

    s = Coordinates(
        phi, theta, rad,
        domain='sph', convention='top_colat')
    return SphericalVoronoi(s)


@pytest.fixture
def filter():
    coeff = np.array([[[1, 0, 0], [1, 0, 0]]])
    return fo.Filter(coefficients=coeff, state=state, comment='my comment')


@pytest.fixture
def filterIIR():
    coeff = np.array([[1, 1/2, 0], [1, 0, 0]])
    return fo.FilterIIR(coeff, sampling_rate=2*np.pi)


@pytest.fixture
def filterFIR():
    coeff = np.array([
        [1, 1/2, 0],
        [1, 1/4, 1/8]])
    desired = np.array([
        [[1, 1/2, 0], [1, 0, 0]],
        [[1, 1/4, 1/8], [1, 0, 0]]
        ])
    return fo.FilterFIR(coeff, sampling_rate=2*np.pi)


@pytest.fixture
def filterSOS():
    sos = np.array([[1, 1/2, 0, 1, 0, 0]])
    return fo.FilterSOS(sos, sampling_rate=2*np.pi)
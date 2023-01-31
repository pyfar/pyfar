import numpy as np
import numpy.testing as npt
import pytest

from pyfar import SamplingSphere
from pyfar.classes.coordinates import sph2cart, cart2sph, cyl2cart


def test_init():
    """Test initialization of empty SamplingSphere object."""
    coords = SamplingSphere()
    assert isinstance(coords, SamplingSphere)


def test_init_comment():
    """Test initialization of empty SamplingSphere object."""
    coords = SamplingSphere(comment='hallo')
    assert isinstance(coords, SamplingSphere)
    assert coords.comment == 'hallo'


def test_init_sh_order():
    """Test initialization of empty SamplingSphere object."""
    coords = SamplingSphere(sh_order=1)
    assert isinstance(coords, SamplingSphere)
    assert coords.sh_order == 1


def test_init_weights():
    """Test initialization of empty SamplingSphere object."""
    coords = SamplingSphere([1, 2], 2, 3, weights=[5, 3])
    assert isinstance(coords, SamplingSphere)
    npt.assert_array_almost_equal(coords.weights, np.array([5, 3]))


@pytest.mark.parametrize('x', [0, 1, -1.])
@pytest.mark.parametrize('y', [0, 1, -1.])
@pytest.mark.parametrize('z', [0, 1, -1.])
def test_init_from_cartesian(x, y, z):
    coords = SamplingSphere.from_cartesian(x, y, z)
    npt.assert_allclose(coords._x, x)
    npt.assert_allclose(coords._y, y)
    npt.assert_allclose(coords._z, z)


@pytest.mark.parametrize('x', [0, 1, -1.])
@pytest.mark.parametrize('y', [0, 1, -1.])
@pytest.mark.parametrize('z', [0, 1, -1.])
@pytest.mark.parametrize('weights', [1])
@pytest.mark.parametrize('comment', ['0'])
def test_init_from_cartesian_with(x, y, z, weights, comment):
    coords = SamplingSphere.from_cartesian(x, y, z, weights, comment)
    npt.assert_allclose(coords._x, x)
    npt.assert_allclose(coords._y, y)
    npt.assert_allclose(coords._z, z)
    coords.comment == comment
    coords.weights == weights


@pytest.mark.parametrize('x', [0, 1, -1.])
@pytest.mark.parametrize('y', [0, 1, -1.])
@pytest.mark.parametrize('z', [0, 1, -1.])
def test_init_from_spherical_colatitude(x, y, z):
    theta, phi, rad = cart2sph(x, y, z)
    coords = SamplingSphere.from_spherical_colatitude(theta, phi, rad)
    # use atol here because of numerical rounding issues introduced in
    # the coordinate conversion
    npt.assert_allclose(coords._x, x, atol=1e-15)
    npt.assert_allclose(coords._y, y, atol=1e-15)
    npt.assert_allclose(coords._z, z, atol=1e-15)


@pytest.mark.parametrize('x', [0, 1, -1.])
@pytest.mark.parametrize('y', [0, 1, -1.])
@pytest.mark.parametrize('z', [0, 1, -1.])
@pytest.mark.parametrize('weights', [1])
@pytest.mark.parametrize('comment', ['0'])
def test_init_from_spherical_colatitude_with(x, y, z, weights, comment):
    theta, phi, rad = cart2sph(x, y, z)
    coords = SamplingSphere.from_spherical_colatitude(
        theta, phi, rad, weights, comment)
    # use atol here because of numerical rounding issues introduced in
    # the coordinate conversion
    npt.assert_allclose(coords._x, x, atol=1e-15)
    npt.assert_allclose(coords._y, y, atol=1e-15)
    npt.assert_allclose(coords._z, z, atol=1e-15)
    coords.comment == comment
    coords.weights == weights


@pytest.mark.parametrize('azimuth', [0, np.pi, -np.pi])
@pytest.mark.parametrize('elevation', [0, np.pi, -np.pi])
@pytest.mark.parametrize('radius', [0, 1, -1.])
@pytest.mark.parametrize('weights', [1])
@pytest.mark.parametrize('comment', ['0'])
def test_init_from_spherical_elevation_with(
        azimuth, elevation, radius, weights, comment):
    coords = SamplingSphere.from_spherical_elevation(
        azimuth, elevation, radius, weights, comment)
    # use atol here because of numerical rounding issues introduced in
    # the coordinate conversion
    x, y, z = sph2cart(azimuth, np.pi / 2 - elevation, radius)
    npt.assert_allclose(coords._x, x, atol=1e-15)
    npt.assert_allclose(coords._y, y, atol=1e-15)
    npt.assert_allclose(coords._z, z, atol=1e-15)
    coords.comment == comment
    coords.weights == weights


@pytest.mark.parametrize('lateral', [0, np.pi, -np.pi])
@pytest.mark.parametrize('polar', [0, np.pi, -np.pi])
@pytest.mark.parametrize('radius', [0, 1, -1.])
@pytest.mark.parametrize('weights', [1])
@pytest.mark.parametrize('comment', ['0'])
def test_init_from_spherical_side_with(
        lateral, polar, radius, weights, comment):
    coords = SamplingSphere.from_spherical_side(
        lateral, polar, radius, weights, comment)
    # use atol here because of numerical rounding issues introduced in
    # the coordinate conversion
    x, z, y = sph2cart(polar, np.pi / 2 - lateral, radius)
    npt.assert_allclose(coords._x, x, atol=1e-15)
    npt.assert_allclose(coords._y, y, atol=1e-15)
    npt.assert_allclose(coords._z, z, atol=1e-15)
    coords.comment == comment
    coords.weights == weights


@pytest.mark.parametrize('phi', [0, np.pi, -np.pi])
@pytest.mark.parametrize('theta', [0, np.pi, -np.pi])
@pytest.mark.parametrize('radius', [0, 1, -1.])
@pytest.mark.parametrize('weights', [1])
@pytest.mark.parametrize('comment', ['0'])
def test_init_from_spherical_front_with(
        phi, theta, radius, weights, comment):
    coords = SamplingSphere.from_spherical_front(
        phi, theta, radius, weights, comment)
    # use atol here because of numerical rounding issues introduced in
    # the coordinate conversion
    y, z, x = sph2cart(phi, theta, radius)
    npt.assert_allclose(coords._x, x, atol=1e-15)
    npt.assert_allclose(coords._y, y, atol=1e-15)
    npt.assert_allclose(coords._z, z, atol=1e-15)
    coords.comment == comment
    coords.weights == weights


@pytest.mark.parametrize('azimuth', [0, np.pi, -np.pi])
@pytest.mark.parametrize('z', [0, 1, -1.])
@pytest.mark.parametrize('rho', [0, 1, -1.])
@pytest.mark.parametrize('weights', [1])
@pytest.mark.parametrize('comment', ['0'])
def test_init_from_cylindrical_with(
        azimuth, z, rho, weights, comment):
    coords = SamplingSphere.from_cylindrical(
        azimuth, z, rho, weights, comment)
    # use atol here because of numerical rounding issues introduced in
    # the coordinate conversion
    x, y, z = cyl2cart(azimuth, z, rho)
    npt.assert_allclose(coords._x, x, atol=1e-15)
    npt.assert_allclose(coords._y, y, atol=1e-15)
    npt.assert_allclose(coords._z, z, atol=1e-15)
    coords.comment == comment
    coords.weights == weights


@pytest.mark.parametrize('azimuth', [0, np.pi, -np.pi])
@pytest.mark.parametrize('elevation', [0, np.pi, -np.pi])
@pytest.mark.parametrize('radius', [0, 1, -1.])
def test_init_from_spherical_elevation(azimuth, elevation, radius):
    coords = SamplingSphere.from_spherical_elevation(
        azimuth, elevation, radius)
    # use atol here because of numerical rounding issues introduced in
    # the coordinate conversion
    x, y, z = sph2cart(azimuth, np.pi / 2 - elevation, radius)
    npt.assert_allclose(coords._x, x, atol=1e-15)
    npt.assert_allclose(coords._y, y, atol=1e-15)
    npt.assert_allclose(coords._z, z, atol=1e-15)


@pytest.mark.parametrize('lateral', [0, np.pi, -np.pi])
@pytest.mark.parametrize('polar', [0, np.pi, -np.pi])
@pytest.mark.parametrize('radius', [0, 1, -1.])
def test_init_from_spherical_side(lateral, polar, radius):
    coords = SamplingSphere.from_spherical_side(lateral, polar, radius)
    # use atol here because of numerical rounding issues introduced in
    # the coordinate conversion
    x, z, y = sph2cart(polar, np.pi / 2 - lateral, radius)
    npt.assert_allclose(coords._x, x, atol=1e-15)
    npt.assert_allclose(coords._y, y, atol=1e-15)
    npt.assert_allclose(coords._z, z, atol=1e-15)


@pytest.mark.parametrize('phi', [0, np.pi, -np.pi])
@pytest.mark.parametrize('theta', [0, np.pi, -np.pi])
@pytest.mark.parametrize('radius', [0, 1, -1.])
def test_init_from_spherical_front(phi, theta, radius):
    coords = SamplingSphere.from_spherical_front(phi, theta, radius)
    # use atol here because of numerical rounding issues introduced in
    # the coordinate conversion
    y, z, x = sph2cart(phi, theta, radius)
    npt.assert_allclose(coords._x, x, atol=1e-15)
    npt.assert_allclose(coords._y, y, atol=1e-15)
    npt.assert_allclose(coords._z, z, atol=1e-15)


@pytest.mark.parametrize('azimuth', [0, np.pi, -np.pi])
@pytest.mark.parametrize('z', [0, 1, -1.])
@pytest.mark.parametrize('radius_z', [0, 1, -1.])
def test_init_from_cylindrical(azimuth, z, radius_z):
    coords = SamplingSphere.from_cylindrical(azimuth, z, radius_z)
    # use atol here because of numerical rounding issues introduced in
    # the coordinate conversion
    x, y, z = cyl2cart(azimuth, z, radius_z)
    npt.assert_allclose(coords._x, x, atol=1e-15)
    npt.assert_allclose(coords._y, y, atol=1e-15)
    npt.assert_allclose(coords._z, z, atol=1e-15)

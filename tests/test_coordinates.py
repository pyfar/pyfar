import numpy as np
import numpy.testing as npt
import pytest
from pytest import raises
import pandas as pd

import haiopy
from haiopy import Coordinates
import haiopy.coordinates as coordinates


# %% Test module functions ----------------------------------------------------

# TODO: Do I have to provide an error string to assert or does pytest show more
#       detailed information?
def test_coordinate_systems():
    systems, coords = coordinates._coordinate_systems()

    # check object type
    assert isinstance(systems, dict)
    assert isinstance(coords, dict)

    # check completeness of systems
    for domain in systems:
        for convention in systems[domain]:
            assert "description_short" in systems[domain][convention], \
                "{} ({}) is missing entry 'description_short'".format(domain, convention)
            assert "coordinates"       in systems[domain][convention], \
                "{} ({}) is missing entry 'coordinates'".format(domain, convention)
            assert "units"             in systems[domain][convention], \
                "{} ({}) is missing entry 'units'".format(domain, convention)
            assert "description"       in systems[domain][convention], \
                "{} ({}) is missing entry 'description'".format(domain, convention)

    # check if units agree across coordinates that appear more than once
    for coord in coords:
         # get unique first entry
        units     = coords[coord]['units'].copy()
        units_ref = pd.unique(units[0])
        for cc in range(1, len(units)):
            # get nex entry for comparison
            units_test = pd.unique(units[cc])
            # compare
            assert all(units_ref == units_test), \
                "'{}' has units {} in {} ({}) but units {} in {} ({})".\
                    format(coord, units_ref, \
                           coords[coord]['domain'][0], \
                           coords[coord]['convention'][0], \
                           units_test, \
                           coords[coord]['domain'][cc], \
                           coords[coord]['convention'][cc])

def test_exist_coordinate_systems():

    # things that should pass
    coordinates._exist_coordinate_systems
    coordinates._exist_coordinate_systems('sph')
    coordinates._exist_coordinate_systems('sph', 'side')

    # things that shoud not pass
    with raises(AssertionError):
         coordinates._exist_coordinate_systems(None, 'side')
    with raises(AssertionError):
         coordinates._exist_coordinate_systems('shp')
    with raises(AssertionError):
         coordinates._exist_coordinate_systems('sph', 'tight')


# %% Test Coordinates() class ------------------------------------------------

# def test_coordinates_init():
#     coords = Coordinates()
#     assert isinstance(coords, Coordinates)

# def test_coordinates_init_val():

#     coords = Coordinates(1, 0, 0)
#     assert isinstance(coords, Coordinates)

# def test_coordinates_init_incomplete():
#     x = [1, 2]
#     y = 1
#     z = 1
#     with raises(ValueError):
#         Coordinates(x, y, z)
#         pytest.fail("Input arrays need to have same dimensions.")

# def test_coordinates_init_from_cartesian():
#     x = 1
#     y = 0
#     z = 0
#     coords = Coordinates.from_cartesian(x, y, z)
#     npt.assert_allclose(coords._x, x)
#     npt.assert_allclose(coords._y, y)
#     npt.assert_allclose(coords._z, z)


# def test_coordinates_init_from_spherical():
#     x = 1
#     y = 0
#     z = 0
#     rad, theta, phi = _cart2sph(x, y, z)
#     coords = Coordinates.from_spherical(rad, theta, phi)
#     # use atol here because of numerical rounding issues introduced in
#     # the coordinate conversion
#     npt.assert_allclose(coords._x, x, atol=1e-15)
#     npt.assert_allclose(coords._y, y, atol=1e-15)
#     npt.assert_allclose(coords._z, z, atol=1e-15)

# def test_coordinates_init_from_array_spherical():
#     rad = [1., 1., 1., 1.]
#     ele = [np.pi/2, np.pi/2, 0, np.pi/2]
#     azi = [0, np.pi/2, 0, np.pi/4]

#     points = np.array([rad, ele, azi])
#     coords = Coordinates.from_array(points, coordinate_system='spherical')

#     npt.assert_allclose(coords.radius, rad, atol=1e-15)
#     npt.assert_allclose(coords.elevation, ele, atol=1e-15)
#     npt.assert_allclose(coords.azimuth, azi, atol=1e-15)

# def test_coordinates_init_from_array_cartesian():
#     x = [1, 0, 0, 0]
#     y = [0, 1, 0, 0]
#     z = [0, 0, 1, 0]

#     points = np.array([x, y, z])
#     coords = Coordinates.from_array(points)

#     npt.assert_allclose(coords._x, x, atol=1e-15)
#     npt.assert_allclose(coords._y, y, atol=1e-15)
#     npt.assert_allclose(coords._z, z, atol=1e-15)

# def test_getter_x():
#     x = np.array([1, 0], dtype=np.double)
#     coords = Coordinates()
#     coords._x = x
#     npt.assert_allclose(coords.x, x)

# def test_getter_y():
#     y = np.array([1, 0], dtype=np.double)
#     coords = Coordinates()
#     coords._y = y
#     npt.assert_allclose(coords.y, y)

# def test_getter_z():
#     z = np.array([1, 0], dtype=np.double)
#     coords = Coordinates()
#     coords._z = z
#     npt.assert_allclose(coords.z, z)

# def test_setter_x():
#     value = np.array([1.0, 1], dtype=np.double)
#     coords = Coordinates()
#     coords.x = value
#     npt.assert_allclose(value, coords._x)

# def test_setter_y():
#     value = np.array([1.0, 1], dtype=np.double)
#     coords = Coordinates()
#     coords.y = value
#     npt.assert_allclose(value, coords._y)

# def test_setter_z():
#     value = np.array([1.0, 1], dtype=np.double)
#     coords = Coordinates()
#     coords.z = value
#     npt.assert_allclose(value, coords._z)

# def test_getter_ele():
#     value = np.pi/2
#     coords = Coordinates()
#     coords.z = 0
#     coords.y = 0
#     coords.x = 1
#     npt.assert_allclose(coords.elevation, value)

# def test_getter_radius():
#     value = 1
#     coords = Coordinates()
#     coords.z = 0
#     coords.y = 1
#     coords.x = 0
#     npt.assert_allclose(coords.radius, value)

# def test_getter_azi():
#     azi = np.pi/2
#     coords = Coordinates()
#     coords.z = 0
#     coords.y = 1
#     coords.x = 0
#     npt.assert_allclose(coords.azimuth, azi)

# def test_setter_rad():
#     eps = np.spacing(1)
#     rad = 0.5
#     x = 0.5
#     y = 0
#     z = 0
#     coords = Coordinates(1, 0, 0)
#     coords.radius = rad
#     npt.assert_allclose(coords._x, x, atol=eps)
#     npt.assert_allclose(coords._y, y, atol=eps)
#     npt.assert_allclose(coords._z, z, atol=eps)

# def test_setter_ele():
#     eps = np.spacing(1)
#     ele = 0
#     x = 0
#     y = 0
#     z = 1
#     coords = Coordinates(1, 0, 0)
#     coords.elevation = ele
#     npt.assert_allclose(coords._x, x, atol=eps)
#     npt.assert_allclose(coords._y, y, atol=eps)
#     npt.assert_allclose(coords._z, z, atol=eps)


# def test_setter_azi():
#     eps = np.spacing(1)
#     azi = np.pi/2
#     x = 0
#     y = 1
#     z = 0
#     coords = Coordinates(1, 0, 0)
#     coords.azimuth = azi
#     npt.assert_allclose(coords._x, x, atol=eps)
#     npt.assert_allclose(coords._y, y, atol=eps)
#     npt.assert_allclose(coords._z, z, atol=eps)

# def test_getter_latitude():
#     x = 1
#     y = 0
#     z = 0.5

#     height, lat, lon = _cart2latlon(x, y, z)
#     coords = Coordinates(x, y, z)
#     npt.assert_allclose(coords.latitude, lat)

# def test_getter_longitude():
#     x = 1
#     y = 0
#     z = 0.5

#     height, lat, lon = _cart2latlon(x, y, z)
#     coords = Coordinates(x, y, z)
#     npt.assert_allclose(coords.longitude, lon)

# def test_getter_cartesian():
#     x = [1, 0, 0, 0]
#     y = [0, 1, 0, 0]
#     z = [0, 0, 1, 0]

#     coords = Coordinates(x, y, z)
#     ref = np.vstack((x, y, z))
#     npt.assert_allclose(coords.cartesian, ref)


# def test_setter_cartesian():
#     x = np.array([1, 0, 0, 0])
#     y = np.array([0, 1, 0, 0])
#     z = np.array([0, 0, 1, 0])
#     cart = np.vstack((x, y, z))
#     coords = Coordinates()
#     coords.cartesian = cart
#     npt.assert_allclose(coords.cartesian, cart)


# def test_getter_spherical():
#     x = np.array([1, 0, 0, 1], dtype=np.float64)
#     y = np.array([0, 1, 0, 1], dtype=np.float64)
#     z = np.array([0, 0, 1, 1], dtype=np.float64)

#     rad, theta, phi = _cart2sph(x, y, z)

#     coords = Coordinates(x, y, z)
#     ref = np.vstack((rad, theta, phi))
#     npt.assert_allclose(coords.spherical, ref)


# def test_setter_spherical():
#     eps = np.spacing(1)
#     x = np.array([1, 0, 0, 1], dtype=np.float64)
#     y = np.array([0, 1, 0, 1], dtype=np.float64)
#     z = np.array([0, 0, 1, 1], dtype=np.float64)
#     rad, theta, phi = _cart2sph(x, y, z)
#     spherial = np.vstack((rad, theta, phi))
#     coords = Coordinates()
#     coords.spherical = spherial
#     npt.assert_allclose(coords._x, x, atol=eps)
#     npt.assert_allclose(coords._y, y, atol=eps)
#     npt.assert_allclose(coords._z, z, atol=eps)


# def test_n_points():
#     coords = Coordinates([1, 0], [1, 1], [0, 1])
#     assert coords.n_points == 2


# def test_find_nearest():
#     coords = Coordinates([1, 0], [1, 1], [0, 1])
#     point = Coordinates(1, 1, 0)

#     dist, idx = coords.find_nearest_point(point)

#     assert idx == 0


# def test_len():
#     coords = Coordinates([1, 0], [1, 1], [0, 1])
#     assert len(coords) == 2


# def test_getitem():
#     coords = Coordinates([1, 0], [1, 1], [0, 1])
#     getcoords = coords[0]
#     npt.assert_allclose(np.squeeze(getcoords.cartesian), np.array([1, 1, 0]))


# def test_setitem():
#     coords = Coordinates([0, 0], [1, 1], [0, 1])
#     setcoords = Coordinates(1, 1, 0)
#     coords[0] = setcoords
#     npt.assert_allclose(np.squeeze(coords.cartesian),
#                         np.array([[1, 0], [1, 1], [0, 1]]))

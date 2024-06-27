import numpy as np
import numpy.testing as npt
import pytest

import pyfar as pf


def test_find_nearest_simple():
    """Test returns of find_nearest_k"""
    # 1D cartesian, nearest point
    x = np.arange(6)
    coords = pf.Coordinates(x, 0, 0)
    find = pf.Coordinates(1, 0, 0)
    i, d = coords.find_nearest(find)
    assert i[0] == 1
    assert find == coords[i]
    assert d == 0


@pytest.mark.parametrize('azimuth,distance_measure,distance', [
    (0, 'spherical_radians', 0),                        # radians match
    (np.pi / 16, 'spherical_radians', np.pi / 16),      # radians no match
    (0, 'spherical_meter', 0),                          # meters match
    (np.pi / 16, 'spherical_meter', np.pi / 16 * 1.1)   # meters no match
])
def test_find_nearest_simple_spherical_distance(
        azimuth, distance_measure, distance):
    """Test spherical distance measures for find nearest"""
    # 1D spherical coordinates points in front and to the left
    coords = pf.Coordinates().from_spherical_elevation([0, np.pi/2], 0, 1.1)
    find = pf.Coordinates().from_spherical_elevation(azimuth, 0, 1.1)
    i, d = coords.find_nearest(find, distance_measure=distance_measure)
    assert i[0] == 0
    assert np.abs(d - distance) < 1e-15


def test_find_nearest_1d_2d():
    # 1D spherical, nearest point
    coords = pf.Coordinates(np.arange(10), 0, 1)

    find = pf.Coordinates(
        np.array([[0, 1], [2, 3]]), 0, 1)
    i, d = coords.find_nearest(find)
    assert find == coords[i]
    npt.assert_equal(i[0].shape, d.shape)
    npt.assert_equal(i[0].shape, (2, 2))


def test_find_nearest_2d_2d():
    # 1D spherical, nearest point
    x = np.arange(6*7).reshape((6, 7))
    coords = pf.Coordinates(x, 0, 1)

    find = pf.Coordinates(
        np.array([[0, 1], [2, 3], [4, 5]]), 0, 1)
    i, d = coords.find_nearest(find)
    assert coords[i] == find
    npt.assert_equal(i[0].shape, d.shape)
    npt.assert_equal(d.shape, (3, 2))
    npt.assert_equal(len(i), 2)


def test_find_nearest_2d_2d_k5():
    # 1D spherical, nearest point
    x = np.arange(9).reshape((3, 3))
    coords = pf.Coordinates(x, 0, 1)

    find = pf.Coordinates(
        np.array([[0, 1], [2, 3]]), 0, 1)
    i, d = coords.find_nearest(find, k=5)
    assert coords[i[0]] == find  # k=0
    npt.assert_equal(d.shape, (5, 2, 2))
    npt.assert_equal(len(i), 5)
    npt.assert_equal(len(i[0]), 2)
    npt.assert_equal(i[0][0].shape, (2, 2))


def test_find_nearest_2d_k3():
    # 1D spherical, nearest point
    coords = pf.samplings.sph_gaussian(sh_order=47)
    k = 5
    find = pf.Coordinates.from_spherical_elevation(
        np.array([[0, np.pi/2], [np.pi, 3*np.pi/2]]), 0, 1)
    i, d = coords.find_nearest(find, k=k)
    npt.assert_equal(i[0][0].shape, (2, 2))
    npt.assert_equal(len(i), k)
    npt.assert_equal(d.shape, (k, 2, 2))
    for kk in range(k):
        actual_distance = np.sqrt(np.sum(
            (coords[i[kk]].cartesian - find.cartesian)**2, axis=-1))
        npt.assert_equal(actual_distance, d[kk])


def test_find_nearest_error():
    coords = pf.samplings.sph_gaussian(sh_order=47)
    find = pf.Coordinates(1, 0, 0)

    # test out of range parameters
    with pytest.raises(ValueError):
        coords.find_nearest(find, -1)

    # test Coordinate object as input
    with pytest.raises(ValueError):
        coords.find_nearest(5, 1)

    # test wrong string for distance measure
    with pytest.raises(ValueError):
        coords.find_nearest(find, 1, 'bla')

    # test wrong type for distance measure
    with pytest.raises(ValueError):
        coords.find_nearest(find, 1, 5)

    # test negative radius_tol
    with pytest.raises(ValueError,
                       match="radius_tol must be a non negative number."):
        coords.find_nearest(find, 1, "spherical_radians", -1)


def test_find_nearest_radius_tol():
    coords = pf.Coordinates.from_spherical_elevation(
        np.arange(0, 360, 10)*np.pi/180, 0, 1)
    radius = coords.radius
    radius[0] = 1.09
    coords.radius = radius
    coords2find = pf.Coordinates.from_spherical_elevation(0, 0, 1)
    idx, _ = coords.find_nearest(
        coords2find,
        1,
        "spherical_radians",
        radius_tol=0.1)
    assert coords[idx] == coords[0]


def test_find_within_simple():
    x = np.arange(6)
    coords = pf.Coordinates(x, 0, 0)
    find = pf.Coordinates(2, 0, 0)
    index = coords.find_within(find, 1)
    assert len(index) == 1
    npt.assert_equal(index[0], [1, 2, 3])


def test_find_within_multiple_points():
    x = np.arange(6)
    coords = pf.Coordinates(x, 0, 0)
    find = pf.Coordinates([2, 3], 0, 0)
    index = coords.find_within(find, 1)
    assert len(index) == find.csize
    for i in range(find.csize):
        index_desired = coords.find_within(find[i], 1)
        npt.assert_equal(index[i], index_desired)


def test_find_within_multiple_dim_points():
    x = np.arange(9).reshape((3, 3))
    coords = pf.Coordinates(x, 0, 0)
    find = pf.Coordinates([2, 0], 0, 0)
    index = coords.find_within(find, 1)
    assert len(index) == find.csize
    for i in range(find.csize):
        index_desired = coords.find_within(find[i], 1)
        assert coords[index[i]] == coords[index_desired]


def test_find_within_error():
    coords = pf.samplings.sph_gaussian(sh_order=47)
    find = pf.Coordinates(1, 0, 0)

    # test out of range parameters
    with pytest.raises(ValueError):
        coords.find_within(find, -1, 'euclidean')

    # test Coordinate object as input
    with pytest.raises(ValueError):
        coords.find_within(5, 1)

    # test wrong string for distance measure
    with pytest.raises(ValueError):
        coords.find_within(find, 1, 'bla')

    # test wrong type for distance measure
    with pytest.raises(ValueError):
        coords.find_within(find, 1, 5)

    with pytest.raises(ValueError):
        coords.find_within(find, 1, atol=-1)

    with pytest.raises(ValueError):
        coords.find_within(find, 1, atol='h')

    with pytest.raises(ValueError):
        coords.find_within(find, 1, radius_tol='h')

    with pytest.raises(ValueError):
        coords.find_within(find, 1, return_sorted=-1)


@pytest.mark.parametrize('distance_measure', [
     'spherical_radians', 'spherical_meter'
])
@pytest.mark.parametrize('radius', [.5, 1, 2])
def test_find_within_spherical(distance_measure, radius):
    '''Test spherical distance measures for different radii'''
    # Sampling grid in the median plane
    coords = pf.Coordinates.from_spherical_front(
        np.arange(0, 360, 10)*np.pi/180, np.pi/2, radius)
    # query point at north pole
    find = pf.Coordinates(0, 0, radius)
    # search within 90 degrees
    distance = np.pi / 2
    if distance_measure == 'spherical_meter':
        distance *= radius
    spatial_mask = coords.find_within(
        find,
        distance=distance,
        distance_measure=distance_measure)
    # all points with positive z-coordinates must be found
    npt.assert_array_equal(coords[spatial_mask], coords[coords.z >= 0])


def test_find_within_atol():
    coords = pf.Coordinates(
        np.arange(0, 1, 0.1), 0, 0)
    find = pf.Coordinates(0.5, 0, 0)
    spatial_mask = coords.find_within(
        find,
        distance=0.1,
        distance_measure='euclidean',
        atol=0.11)
    npt.assert_array_equal(coords[spatial_mask].csize, 5)

    spatial_mask = coords.find_within(
        find,
        distance=0.1,
        distance_measure='euclidean',
        atol=0.05)
    npt.assert_array_equal(coords[spatial_mask].csize, 3)


def test_find_within_tol_radius():
    '''Test spherical distance measure with tolerance for radius'''
    # Sampling grid in the median plane with varying radii
    coords = pf.Coordinates.from_spherical_front(
        np.arange(0, 360, 10)*np.pi/180, np.pi/2, 1)
    radius = coords.radius
    radius[0] = 1.01
    coords.radius = radius
    # query point at north pole
    find = pf.Coordinates(0, 0, 1)
    # search within 90 degrees
    spatial_mask = coords.find_within(
        find,
        distance=np.pi/2,
        distance_measure='spherical_radians',
        radius_tol=0.011)
    # all points with positive z-coordinates must be found
    npt.assert_array_equal(coords[spatial_mask], coords[coords.z >= 0])

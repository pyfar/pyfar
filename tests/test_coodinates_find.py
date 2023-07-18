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
    assert i == 1
    assert d == 0


def test_find_nearest_2d():
    # 1D spherical, nearest point
    coords = pf.samplings.sph_gaussian(sh_order=47)

    find = pf.Coordinates.from_spherical_elevation(
        np.array([[0, np.pi/2], [np.pi, 3*np.pi/2]]), 0, 1)
    i, d = coords.find_nearest(find)
    assert i == 1
    assert d == 0


def test_find_nearest_2d_k3():
    # 1D spherical, nearest point
    coords = pf.samplings.sph_gaussian(sh_order=47)
    k = 5
    find = pf.Coordinates.from_spherical_elevation(
        np.array([[0, np.pi/2], [np.pi, 3*np.pi/2]]), 0, 1)
    i, d = coords.find_nearest(find, k=k)
    npt.assert_equal(i.shape, d.shape)
    npt.assert_equal(i.shape, (k, 2, 2))
    for ii in range(k):
        actual_distance = np.sqrt(np.sum(
            (coords[i].cartesian[ii] - find.cartesian)**2, axis=-1))
        npt.assert_equal(actual_distance, d[ii])


def test_find_nearest():
    # 1D cartesian, two nearest points
    i, m = coords.find_nearest(1.2, 0, 0, 2)
    npt.assert_allclose(i, np.array([1, 2]))
    npt.assert_allclose(m, np.array([0, 1, 1, 0, 0, 0]))

    # 1D cartesian query two points
    i, m = coords.find_nearest([1, 2], 0, 0)
    npt.assert_allclose(i, [1, 2])
    npt.assert_allclose(m, np.array([0, 1, 1, 0, 0, 0]))

    # 2D cartesian, nearest point
    coords = pf.Coordinates(x.reshape(2, 3), 0, 0)
    i, m = coords.find_nearest(1, 0, 0)
    assert i == 1
    npt.assert_allclose(m, np.array([[0, 1, 0], [0, 0, 0]]))

    # test out of range parameters
    with pytest.raises(AssertionError):
        coords.find_nearest_k(1, 0, 0, -1)

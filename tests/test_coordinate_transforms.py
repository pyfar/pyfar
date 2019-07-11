""" Tests for coordinate transforms """

import pytest
import numpy as np
import haiopy.coordinates as coordinates


def test_sph2cart():
    rad, theta, phi = 1, np.pi/2, 0
    x, y, z = coordinates._sph2cart(rad, theta, phi)
    (1, 0, 0) == pytest.approx((x, y, z), abs=2e-16, rel=2e-16)


def test_sph2cart_array():
    rad = np.ones(6)
    theta = np.array([np.pi/2, np.pi/2, np.pi/2, np.pi/2, 0, np.pi])
    phi = np.array([0, np.pi, np.pi/2, np.pi*3/2, 0, 0])
    x, y, z = coordinates._sph2cart(rad, theta, phi)

    xx = np.array([1, -1, 0, 0, 0, 0])
    yy = np.array([0, 0, 1, -1, 0, 0])
    zz = np.array([0, 0, 0, 0, 1, -1])

    np.testing.assert_allclose(xx, x, atol=1e-15)
    np.testing.assert_allclose(yy, y, atol=1e-15)
    np.testing.assert_allclose(zz, z, atol=1e-15)


def test_cart2sph_array():
    x = np.array([1, -1, 0, 0, 0, 0])
    y = np.array([0, 0, 1, -1, 0, 0])
    z = np.array([0, 0, 0, 0, 1, -1])

    rr, tt, pp = coordinates._cart2sph(x, y, z)

    rad = np.ones(6)
    theta = np.array([np.pi/2, np.pi/2, np.pi/2, np.pi/2, 0, np.pi])
    phi = np.array([0, np.pi, np.pi/2, np.pi*3/2, 0, 0])

    np.testing.assert_allclose(rad, rr, atol=1e-15)
    np.testing.assert_allclose(phi, pp, atol=1e-15)
    np.testing.assert_allclose(theta, tt, atol=1e-15)


def test_cart2latlon_array():
    x = np.array([1, -1, 0, 0, 0, 0])
    y = np.array([0, 0, 1, -1, 0, 0])
    z = np.array([0, 0, 0, 0, 1, -1])

    rr, tt, pp = coordinates._cart2latlon(x, y, z)

    rad = np.ones(6)
    theta = np.array([0, 0, 0, 0, np.pi/2, -np.pi/2])
    phi = np.array([0, np.pi, np.pi/2, -np.pi/2, 0, 0])

    np.testing.assert_allclose(rad, rr, atol=1e-15)
    np.testing.assert_allclose(phi, pp, atol=1e-15)
    np.testing.assert_allclose(theta, tt, atol=1e-15)


def test_latlon2cart_array():
    height = np.ones(6)
    theta = np.array([0, 0, 0, 0, np.pi/2, -np.pi/2])
    phi = np.array([0, np.pi, np.pi/2, -np.pi/2, 0, 0])
    xx, yy, zz = coordinates._latlon2cart(height, theta, phi)

    x = np.array([1, -1, 0, 0, 0, 0])
    y = np.array([0, 0, 1, -1, 0, 0])
    z = np.array([0, 0, 0, 0, 1, -1])

    np.testing.assert_allclose(xx, x, atol=1e-15)
    np.testing.assert_allclose(yy, y, atol=1e-15)
    np.testing.assert_allclose(zz, z, atol=1e-15)

import numpy as np
from pyfar import Coordinates
import pyfar.samplings as samplings
import pytest


def test_sph_voronoi():
    dihedral = 2*np.arcsin(np.cos(np.pi/3)/np.sin(np.pi/5))
    R = np.tan(np.pi/3)*np.tan(dihedral/2)
    rho = np.cos(np.pi/5)/np.sin(np.pi/10)

    theta1 = np.arccos((np.cos(np.pi/5)/np.sin(np.pi/5))/np.tan(np.pi/3))

    a2 = 2*np.arccos(rho/R)

    theta2 = theta1+a2
    theta3 = np.pi - theta2
    theta4 = np.pi - theta1

    phi1 = 0
    phi2 = 2*np.pi/3
    phi3 = 4*np.pi/3

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

    verts = np.array([
        [8.72677996e-01, -3.56822090e-01,  3.33333333e-01],
        [3.33333333e-01, -5.77350269e-01,  7.45355992e-01],
        [7.45355992e-01, -5.77350269e-01, -3.33333333e-01],
        [8.72677996e-01,  3.56822090e-01,  3.33333333e-01],
        [-8.72677996e-01, -3.56822090e-01, -3.33333333e-01],
        [-1.27322004e-01, -9.34172359e-01,  3.33333333e-01],
        [-7.45355992e-01, -5.77350269e-01,  3.33333333e-01],
        [1.27322004e-01, -9.34172359e-01, -3.33333333e-01],
        [-3.33333333e-01, -5.77350269e-01, -7.45355992e-01],
        [-8.72677996e-01,  3.56822090e-01, -3.33333333e-01],
        [0.00000000e+00,  0.00000000e+00, -1.00000000e+00],
        [6.66666667e-01, -1.91105568e-16, -7.45355992e-01],
        [7.45355992e-01,  5.77350269e-01, -3.33333333e-01],
        [-3.33333333e-01,  5.77350269e-01, -7.45355992e-01],
        [1.27322004e-01,  9.34172359e-01, -3.33333333e-01],
        [-6.66666667e-01,  2.46373130e-16,  7.45355992e-01],
        [0.00000000e+00,  0.00000000e+00,  1.00000000e+00],
        [3.33333333e-01,  5.77350269e-01,  7.45355992e-01],
        [-1.27322004e-01,  9.34172359e-01,  3.33333333e-01],
        [-7.45355992e-01,  5.77350269e-01,  3.33333333e-01]])

    sv = samplings.SphericalVoronoi(s)
    np.testing.assert_allclose(
        np.sort(np.sum(verts, axis=-1)),
        np.sort(np.sum(sv.vertices, axis=-1)),
        atol=1e-6,
        rtol=1e-6)


def test_weights_from_voronoi():
    s = Coordinates(
        [0, 0, 1, -1, 0, 0], [0, 0, 0, 0, -1, 1], [-1, 1, 0, 0, 0, 0],
        domain='cart', convention='right')

    # test with normalization
    weights = samplings.calculate_sph_voronoi_weights(s, normalize=True)
    desired = np.ones(6)/6
    np.testing.assert_allclose(weights, desired)

    np.testing.assert_allclose(np.sum(weights), 1.)

    # test without normalization
    weights = samplings.calculate_sph_voronoi_weights(s, normalize=False)
    np.testing.assert_allclose(np.sum(weights), 4 * np.pi)


def test_voronoi_error_not_enough_points():
    points = np.random.randn(3, 3)
    points = points/np.linalg.norm(points, axis=0)
    s = Coordinates(points[0], points[1], points[2])
    with pytest.raises(ValueError, match='points needs to be at least 4'):
        samplings.calculate_sph_voronoi_weights(s)


def test___eq___equal(sphericalvoronoi):
    actual = sphericalvoronoi.copy()
    assert sphericalvoronoi == actual


def test___eq___notEqual(sphericalvoronoi):
    actual = sphericalvoronoi.copy()
    actual.center = 42
    assert not sphericalvoronoi == actual

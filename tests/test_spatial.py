import numpy as np
import haiopy.spatial.spatial as spatial
from haiopy.coordinates import Coordinates


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
    phi = np.tile(
        np.array(
            [phi1, phi2, phi3, phi1 + np.pi/3, phi2 + np.pi/3, phi3 + np.pi/3]), 2)
    rad = np.ones(np.size(theta))

    s = Coordinates(
        phi, theta, rad,
        domain='sph', convention='top_colat')

    verts = np.array([
        [ 8.72677996e-01, -3.56822090e-01,  3.33333333e-01],
        [ 3.33333333e-01, -5.77350269e-01,  7.45355992e-01],
        [ 7.45355992e-01, -5.77350269e-01, -3.33333333e-01],
        [ 8.72677996e-01,  3.56822090e-01,  3.33333333e-01],
        [-8.72677996e-01, -3.56822090e-01, -3.33333333e-01],
        [-1.27322004e-01, -9.34172359e-01,  3.33333333e-01],
        [-7.45355992e-01, -5.77350269e-01,  3.33333333e-01],
        [ 1.27322004e-01, -9.34172359e-01, -3.33333333e-01],
        [-3.33333333e-01, -5.77350269e-01, -7.45355992e-01],
        [-8.72677996e-01,  3.56822090e-01, -3.33333333e-01],
        [ 0.00000000e+00,  0.00000000e+00, -1.00000000e+00],
        [ 6.66666667e-01, -1.91105568e-16, -7.45355992e-01],
        [ 7.45355992e-01,  5.77350269e-01, -3.33333333e-01],
        [-3.33333333e-01,  5.77350269e-01, -7.45355992e-01],
        [ 1.27322004e-01,  9.34172359e-01, -3.33333333e-01],
        [-6.66666667e-01,  2.46373130e-16,  7.45355992e-01],
        [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00],
        [ 3.33333333e-01,  5.77350269e-01,  7.45355992e-01],
        [-1.27322004e-01,  9.34172359e-01,  3.33333333e-01],
        [-7.45355992e-01,  5.77350269e-01,  3.33333333e-01]])

    sv = spatial.SphericalVoronoi(s)
    np.testing.assert_allclose(
        np.sort(np.sum(verts, axis=-1)),
        np.sort(np.sum(sv.vertices, axis=-1)),
        atol=1e-6,
        rtol=1e-6)


def test_weights_from_voronoi():
    s = Coordinates(
        [0, 0, 1, -1, 0, 0], [0, 0, 0, 0, -1, 1], [-1, 1, 0, 0, 0, 0],
        domain='cart', convention='right')

    weights = spatial.calculate_sampling_weights_with_spherical_voronoi(s, 10)

    desired = np.ones(6)/6
    np.testing.assert_allclose(weights, desired)

from pyfar import Coordinates
from coordinates import sph2cart, cyl2cart
import numpy as np


class SamplingSphere(Coordinates):
    """Class for samplings on a sphere"""

    def __init__(
            self, x=None, y=None, z=None,
            sh_order=None, weights=None, comment=""):
        """Create a SamplingSphere class object from a set of points in the
        right-handed cartesian coordinate system.

        Parameters
        ----------
        x : ndarray, double
            x-coordinate
        y : ndarray, double
            y-coordinate
        z : ndarray, double
            z-coordinate
        sh_order: int
            the maximum spherical harmonic order
        weights: array like, number, optional
            weighting factors for coordinate points. The `shape` of the array
            must match the `shape` of the individual coordinate arrays.
            The default is ``None``.
        comment : str, optional
            comment about the stored coordinate points. The default is
            ``""``, which initializes an empty string.
        """
        Coordinates.__init__(self, x, y, z, weights=weights, comment=comment)
        if sh_order is not None:
            self._sh_order = int(sh_order)
        else:
            self._sh_order = None

    @classmethod
    def from_cartesian(
            cls, x, y, z, sh_order=None, weights=None, comment: str = ""):
        """Create a SamplingSphere class object from a set of points in the
        right-handed cartesian coordinate system.

        Parameters
        ----------
        x : ndarray, double
            x-coordinate
        y : ndarray, double
            y-coordinate
        z : ndarray, double
            z-coordinate
        sh_order: int
            the maximum spherical harmonic order
        weights: array like, number, optional
            weighting factors for coordinate points. The `shape` of the array
            must match the `shape` of the individual coordinate arrays.
            The default is ``None``.
        comment : str, optional
            comment about the stored coordinate points. The default is
            ``""``, which initializes an empty string.
        """
        return cls(
            x, y, z, sh_order=sh_order, weights=weights, comment=comment)

    @classmethod
    def from_spherical_elevation(
            cls, azimuth, elevation, radius,
            sh_order=None, weights=None, comment: str = ""):
        """Create a SamplingSphere class object from a set of points in the
        spherical coordinate system.

        Parameters
        ----------
        azimuth : ndarray, double
            angle in radiant of rotation from the x-y-plane facing towards
            positive x direction. Used for spherical and cylindrical coordinate
            systems.
        elevation : ndarray, double
            angle in radiant with respect to horizontal plane (x-z-axe).
            Used for spherical coordinate systems.
        radius : ndarray, double
            distance to origin for each point. Used for spherical coordinate
            systems.
        sh_order: int
            the maximum spherical harmonic order
        weights: array like, number, optional
            weighting factors for coordinate points. The `shape` of the array
            must match the `shape` of the individual coordinate arrays.
            The default is ``None``.
        comment : str, optional
            comment about the stored coordinate points. The default is
            ``""``, which initializes an empty string.
        """
        x, y, z = sph2cart(azimuth, np.pi / 2 - elevation, radius)
        return cls(
            x, y, z, sh_order=sh_order, weights=weights, comment=comment)

    @classmethod
    def from_spherical_colatitude(
            cls, azimuth, colatitude, radius,
            sh_order=None, weights=None, comment: str = ""):
        """Create a SamplingSphere class object from a set of points in the
        spherical coordinate system.

        Parameters
        ----------
        azimuth : ndarray, double
            angle in radiant of rotation from the x-y-plane facing towards
            positive x direction. Used for spherical and cylindrical coordinate
            systems.
        colatitude : ndarray, double
            angle in radiant with respect to polar axis (z-axe). Used for
            spherical coordinate systems.
        radius : ndarray, double
            distance to origin for each point. Used for spherical coordinate
            systems.
        sh_order: int
            the maximum spherical harmonic order
        weights: array like, number, optional
            weighting factors for coordinate points. The `shape` of the array
            must match the `shape` of the individual coordinate arrays.
            The default is ``None``.
        comment : str, optional
            comment about the stored coordinate points. The default is
            ``""``, which initializes an empty string.
        """
        x, y, z = sph2cart(azimuth, colatitude, radius)
        return cls(
            x, y, z, sh_order=sh_order, weights=weights, comment=comment)

    @classmethod
    def from_spherical_side(
            cls, lateral, polar, radius,
            sh_order=None, weights=None, comment: str = ""):
        """Create a SamplingSphere class object from a set of points in the
        spherical coordinate system.

        Parameters
        ----------
        lateral : ndarray, double
            angle in radiant with respect to horizontal plane (x-y-axe).
            Used for spherical coordinate systems.
        polar : ndarray, double
            angle in radiant of rotation from the x-z-plane facing towards
            positive x direction. Used for spherical coordinate systems.
        radius : ndarray, double
            distance to origin for each point. Used for spherical coordinate
            systems.
        sh_order: int
            the maximum spherical harmonic order
        weights: array like, number, optional
            weighting factors for coordinate points. The `shape` of the array
            must match the `shape` of the individual coordinate arrays.
            The default is ``None``.
        comment : str, optional
            comment about the stored coordinate points. The default is
            ``""``, which initializes an empty string.
        """
        x, z, y = sph2cart(polar, np.pi / 2 - lateral, radius)
        return cls(
            x, y, z, sh_order=sh_order, weights=weights, comment=comment)

    @classmethod
    def from_spherical_front(
            cls, phi, theta, radius,
            sh_order=None, weights=None, comment: str = ""):
        """Create a SamplingSphere class object from a set of points in the
        spherical coordinate system.

        Parameters
        ----------
        phi : ndarray, double
            Tangle in radiant of rotation from the y-z-plane facing towards
            positive y direction. Used for spherical coordinate systems.
        theta : ndarray, double
            angle in radiant with respect to polar axis (x-axe). Used for
            spherical coordinate systems.
        radius : ndarray, double
            distance to origin for each point. Used for spherical coordinate
            systems.
        sh_order: int
            the maximum spherical harmonic order
        weights: array like, number, optional
            weighting factors for coordinate points. The `shape` of the array
            must match the `shape` of the individual coordinate arrays.
            The default is ``None``.
        comment : str, optional
            comment about the stored coordinate points. The default is
            ``""``, which initializes an empty string.
        """
        y, z, x = sph2cart(phi, theta, radius)
        return cls(
            x, y, z, sh_order=sh_order, weights=weights, comment=comment)

    @classmethod
    def from_cylindrical(
            cls, azimuth, z, rho,
            sh_order=None, weights=None, comment: str = ""):
        """Create a SamplingSphere class object from a set of points in the
        cylindrical coordinate system.

        Parameters
        ----------
        azimuth : ndarray, double
            angle in radiant of rotation from the x-y-plane facing towards
            positive x direction. Used for spherical and cylindrical coordinate
            systems.
        z : ndarray, double
            The z coordinate
        rho : ndarray, double
            distance to origin for each point in the x-y-plane. Used for
            cylindrical coordinate systems.
        sh_order: int
            the maximum spherical harmonic order
        weights: array like, number, optional
            weighting factors for coordinate points. The `shape` of the array
            must match the `shape` of the individual coordinate arrays.
            The default is ``None``.
        comment : str, optional
            comment about the stored coordinate points. The default is
            ``""``, which initializes an empty string.
        """
        x, y, z = cyl2cart(azimuth, z, rho)
        return cls(
            x, y, z, sh_order=sh_order, weights=weights, comment=comment)

    @property
    def sh_order(self):
        """Get the maximum spherical harmonic order."""
        return self._sh_order

    @sh_order.setter
    def sh_order(self, value):
        """Set the maximum spherical harmonic order."""
        self._sh_order = int(value)

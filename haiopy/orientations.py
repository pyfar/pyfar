import numpy as np
import copy

from haiopy.coordinates import Coordinates
import haiopy


class Orientations(object):
    """Container class for orientations in three-dimensional space
    based on scipy.spatial.transform.Rotation and interfaced via
    View/Up vector domain.

    Attributes
    ----------
    view : array like, number
        View vector in looking direction
    up : array like, number
        Up vector, orthogonal to view vector

    Notes
    -----
    The following domains and conversions between them have to be added:
        - Quaternion
        - Roll/Pitch/Yaw angles

    """
    def __init__(
            self,
            views=None,
            ups=None,
            domain='cart',
            convention='right',
            unit=None,
            weights=None,
            sh_order=None,
            comment=None):
        if views is None:
            views = []
        if ups is None:
            ups = []
        self._set_up_vectors(
            views, ups, domain, convention, unit, weights, sh_order, comment)

    def show(self, positions=None):
        """
        Show a quiver plot of the orientation vectors.

        Parameters
        ----------
        
        mask : boolean numpy array, None
            Plot points in black if mask==True and red if mask==False. The
            default is None, in which case all points are plotted in black.

        Returns
        -------
        None.

        """
        if positions is None:
            positions_1 = np.zeros(self.views.cshape)
            positions = Coordinates(positions_1, positions_1, positions_1)
        elif not isinstance(positions, Coordinates):
            raise TypeError("Positions must be of type Coordinates.")
        elif positions.cshape != self.views.cshape:
            raise ValueError("If provided, there must be the same number"
                             "of positions as orientations.")
            
        ax = haiopy.plot.quiver(positions, self.views, color=(1, 0, 0))
        ax = haiopy.plot.quiver(positions, self.ups, ax=ax, color=(0, 1, 0))
        haiopy.plot.quiver(positions, self.rights, ax=ax, color=(0, 0, 1))

    def _set_up_vectors(
            self,
            views,
            ups,
            domain,
            convention,
            unit,
            weights,
            sh_order,
            comment):
        """Check and set view and up vector(s)."""
        views = np.atleast_2d(views).astype(np.float64)
        ups = np.atleast_2d(ups).astype(np.float64)

        # Validate
        if views.shape != ups.shape:
            raise ValueError(
                "There must be the same number of View and Up Vectors.")
        if views.size:
            if views.shape[-1] != 3:
                raise ValueError(
                    "View and Up Vectors must be either empty or 3D.")
            if not (
                    np.all(np.linalg.norm(views, axis=1))
                    and np.all(np.linalg.norm(ups, axis=1))):
                raise ValueError(
                    "If provided, View and Up Vectors must have a length")

        # Cast to Coordinates
        try:
            self.views = Coordinates(
                views[:, 0], views[:, 1], views[:, 2],
                domain=domain, convention=convention, unit=unit,
                weights=weights, sh_order=sh_order, comment=comment)
            self.ups = Coordinates(
                ups[:, 0], ups[:, 1], ups[:, 2],
                domain=domain, convention=convention, unit=unit,
                weights=weights, sh_order=sh_order, comment=comment)
            # Are View and Up vectors perpendicular?
            if not np.allclose(0, np.einsum(
                    'ij,kj->k', self.views.get_cart(), self.ups.get_cart())):
                raise ValueError("View and Up vectors must be perpendicular")
            rights = np.cross(self.views.get_cart(), self.ups.get_cart())
            self.rights = Coordinates(rights[:, 0], rights[:, 1], rights[:, 2])
        except IndexError:
            self.views = Coordinates(
                domain=domain, convention=convention, unit=unit,
                weights=weights, sh_order=sh_order, comment=comment)
            self.ups = copy.deepcopy(views)
            self.rights = copy.deepcopy(views)

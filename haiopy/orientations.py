import numpy as np

from haiopy.coordinates import Coordinates


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
    def __init__(self, views=None, ups=None,
                domain='cart', convention='right', unit=None,
                weights=None, sh_order=None, comment=None):
        if views is None:
            views = []
        if ups is None:
            ups = []
        self._set_views_ups(views, ups,
                domain, convention, unit,
                weights, sh_order, comment)

    def _set_views_ups(self, views, ups,
                domain, convention, unit,
                weights, sh_order, comment):
        """Check and set view and up vector(s)."""
        views = np.atleast_2d(views).astype(np.float64)
        ups = np.atleast_2d(ups).astype(np.float64)

        if views.shape != ups.shape:
            raise ValueError("There must be the same number of \
                View and Up Vectors.")

        # cast to Coordinates
        try:
            views = Coordinates(
                views[:, 0], views[:, 1], views[:, 2],
                domain=domain, convention=convention, unit=unit,
                weights=weights, sh_order=sh_order, comment=comment)
            ups = Coordinates(
                ups[:, 0], ups[:, 1], ups[:, 2],
                domain=domain, convention=convention, unit=unit,
                weights=weights, sh_order=sh_order, comment=comment)
        except IndexError:
            views = Coordinates(
                domain=domain, convention=convention, unit=unit,
                weights=weights, sh_order=sh_order, comment=comment)
            ups = Coordinates(
                domain=domain, convention=convention, unit=unit,
                weights=weights, sh_order=sh_order, comment=comment)
        
        #

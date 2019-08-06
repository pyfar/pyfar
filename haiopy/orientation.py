import numpy as np


class Orientation(object):
    """Container class for orientation in View/Up vector domain.

    Attributes
    ----------
    view : ndarray, double
        View vector in looking direction
    up : ndarray, double
        Up vector, orthogonal to view vector

    Notes
    -----
    The following domains and conversions between them have to be added:
        - Quaternion
        - Roll/Pitch/Yaw angles

    """
    def __init__(self, view=None, up=None):
        """Init coordinates container

        Attributes
        ----------
        view : ndarray, double
            View vector in looking direction
        up : ndarray, double
            Up vector, orthogonal to view vector
        """

        super(Orientation, self).__init__()
        view = np.asarray(view, dtype=np.float64)
        up = np.asarray(up, dtype=np.float64)

        if not np.shape(view) == np.shape(up):
            raise ValueError("Input arrays need to have same dimensions.")

        if not np.isnan(view, up).any():
            if not np.dot(view, up) == 0:
                raise ValueError(
                    "Input arrays need to be orthogonal to each other.")

        self._view = view
        self._up = up

    @property
    def view(self):
        """The view vector"""
        return self._view

    @view.setter
    def view(self, value):
        if not np.dot(value, self._up) == 0:
            raise ValueError(
                "Input arrays need to be orthogonal to each other.")
        self._view = np.asarray(value, dtype=np.float64)

    @property
    def up(self):
        """The Up vector"""
        return self._up

    @up.setter
    def up(self, value):
        if not np.dot(self._view, value) == 0:
            raise ValueError(
                "Input arrays need to be orthogonal to each other.")
        self._up = np.asarray(value, dtype=np.float64)

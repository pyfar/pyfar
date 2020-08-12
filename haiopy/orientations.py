import numpy as np
import copy

from haiopy.coordinates import Coordinates
import haiopy

from scipy.spatial.transform import Rotation


class Orientations(Rotation):


    def __init__(self, quat=None, normalize=True, copy=True):
        if quat is None:
            quat = np.array([0., 0., 0., 1.])
        super().__init__(quat, normalize=normalize, copy=copy)

    @classmethod
    def from_view_up(cls, views, ups):
        try:
            views = views.get_cart()
            ups = ups.get_cart()
        except AttributeError:
            views = np.atleast_2d(views).astype(np.float64)
            ups = np.atleast_2d(ups).astype(np.float64)
        if views.shape != ups.shape:
            raise ValueError(
                "There must be the same number of View and Up Vectors.")
        if views.shape[-1] != 3:
            raise ValueError(f"Expected `views` and `ups` to have shape (N, 3)"
                             " or (3,), got {views.shape}")

        if not (np.all(np.linalg.norm(views, axis=1))
                and np.all(np.linalg.norm(ups, axis=1))):
            raise ValueError("View and Up Vectors must have a length.")
        if not np.allclose(0, np.einsum('ij,kj->k', views, ups)):
            raise ValueError("View and Up vectors must be perpendicular.")
        
        rights = np.cross(views, ups)
        
        rotation_matrix = np.empty((views.shape[0], views.shape[1], 3))
        
        rotation_matrix[:, 0, :3] = views
        rotation_matrix[:, 1, :3] = ups
        rotation_matrix[:, 2, :3] = rights

        return cls.from_matrix(rotation_matrix)


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
            positions_1 = np.zeros((self._quat.shape[0], 3))
            positions = Coordinates(positions_1, positions_1, positions_1)
        elif not isinstance(positions, Coordinates):
            raise TypeError("Positions must be of type Coordinates.")
        elif positions.cshape[0] != self._quat.shape[0]:
            raise ValueError("If provided, there must be the same number"
                             "of positions as orientations.")

        # Create view, up and right vectors from Rotation object
        views, rights, ups = [np.atleast_2d(self.apply(x)) for x in np.eye(3)]
        views, rights, ups = [Coordinates(x[:, 0], x[:, 1], x[:, 2])
                              for x in (views, rights, ups)]

        ax = haiopy.plot.quiver(positions, views, color=(1, 0, 0))
        ax = haiopy.plot.quiver(positions, ups, ax=ax, color=(0, 1, 0))
        haiopy.plot.quiver(positions, rights, ax=ax, color=(0, 0, 1))

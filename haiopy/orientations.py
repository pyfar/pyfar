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

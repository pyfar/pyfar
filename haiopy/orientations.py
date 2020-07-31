import numpy as np


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
    def __init__(self, view=None, up=None):
        pass

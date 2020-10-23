from scipy.spatial.transform import Rotation
import numpy as np

import haiopy


class Orientations(Rotation):
    """Orientations in 3d is a subclass of scipy.spatial.transform.Rotation.

    This container class for Orientations in the three-dimensional space,
    is a subclass of scipy.spatial.transform.Rotation and equally based on
    quaternions of shape (N, 4). It inherits all methods of the Rotation class
    and adds the creation from perpendicular view and up vectors and a
    convenient plot function.

    An orientation can be visualized with the triple of view, up and right
    vectors and it istied to the object's local coordinate system.
    Alternatively the object's orientation can be illustrated with help of the
    right hand: Thumb (view), forefinger (up) and middle finger (right).

    Examples
    --------
    from haiopy.orientations import Orientations

    >>> views = [[1, 0, 0], [2, 0, 0]]
    >>> ups = [[0, 1, 0], [0, -2, 0]]
    >>> orientations = Orientations.from_view_up(views, ups)

    Visualize orientations at certain positions:

    >>> positions = [[0, 0.5, 0], [0, -0.5, 0]]
    >>> orientations.show(positions)

    Rotate first element of orientations:

    >>> from scipy.spatial.transform import Rotation
    >>> rot_x45 = Rotation.from_euler('x', 45, degrees=True)
    >>> orientations[1] = orientations[1] * rot_x45
    >>> orientations.show(positions)

    To create `Orientations` objects use ``from_...`` methods.
    ``Orientations(...)`` is not supposed to be instantiated directly.

    Attributes
    ----------
        quat : array_like, shape (N, 4) or (4,)
            Each row is a (possibly non-unit norm) quaternion in scalar-last
            (x, y, z, w) format. Each quaternion will be normalized to unit
            norm.
    """

    def __init__(self, quat=None, normalize=True, copy=True):
        if quat is None:
            quat = np.array([0., 0., 0., 1.])
        super().__init__(quat, copy=copy)

    @classmethod
    def from_view_up(cls, views, ups):
        """Initialize Orientations from a view an up vector.

        Orientations are internally stored as quaternions for better spherical
        linear interpolation (SLERP) and spherical harmonics operations.
        More intuitionally, they can be expressed as view and up vectors
        which cannot be collinear. In this case, they are restricted to be
        perpendicular to minimize rounding errors.

        Parameters
        ----------
        views : array_like, shape (N, 3) or (3,)
            A single vector or a stack of vectors, giving the look-direction of
            an object in three-dimensional space, e.g. from a listener, or the
            acoustic axis of a loudspeaker, or the direction of a main lobe.

        ups : array_like, shape (N, 3) or (3,)
            A single vector or a stack of vectors, giving the up-direction of
            an object, which is usually the up-direction in world-space.

        Returns
        -------
        orientations : `Orientations` instance
            Object containing the orientations represented by quaternions.
        """
        views = np.atleast_2d(views).astype(np.float64)
        ups = np.atleast_2d(ups).astype(np.float64)

        if (views.ndim > 2 or views.shape[-1] != 3
                or ups.ndim > 2 or ups.shape[-1] != 3):
            raise ValueError(f"Expected `views` and `ups` to have shape (N, 3)"
                             f" or (3,), got {views.shape}")
        if views.shape == ups.shape:
            pass
        elif views.shape[0] > 1 and ups.shape[0] == 1:
            ups = np.repeat(ups, views.shape[0], axis=0)
        elif ups.shape[0] > 1 and views.shape[0] == 1:
            views = np.repeat(views, ups.shape[0], axis=0)
        else:
            raise ValueError("Expected 1:1, 1:N or N:1 `views` and `ups` "
                             f"not M:N, got {views.shape} and {ups.shape}")

        if not (np.all(np.linalg.norm(views, axis=1))
                and np.all(np.linalg.norm(ups, axis=1))):
            raise ValueError("View and Up Vectors must have a length.")
        if not np.allclose(0, np.einsum('ij,kj->k', views, ups)):
            raise ValueError("View and Up vectors must be perpendicular.")

        # Assuming that the direction of the cross product is defined
        # by the right-hand rule
        rights = np.cross(views, ups)
        rotation_matrix = np.asarray([views, ups, rights])
        rotation_matrix = np.swapaxes(rotation_matrix, 0, 1)
        # rotation_matrix = np.empty((views.shape[0], views.shape[1], 3))

        # rotation_matrix[:, 0, :3] = views
        # rotation_matrix[:, 1, :3] = ups
        # rotation_matrix[:, 2, :3] = rights

        return super().from_matrix(rotation_matrix)

    def show(self, positions=None,
             show_views=True, show_ups=True, show_rights=True, **kwargs):
        """
        Visualize Orientations as triples of view (red), up (green) and
        right (blue) vectors in a quiver plot.

        Parameters
        ----------
        positions : array_like, shape (O, 3), O is len(self)
            These are the positions of each vector triple. If not provided,
            all triples are positioned in the origin of the coordinate system.
        show_views: bool
            select wether to show the view vectors or not.
            The default is True.
        show_ups: bool
            select wether to show the up vectors or not.
            The default is True.
        show_rights: bool
            select wether to show the right vectors or not.
            The default is True.

        Returns
        -------
        ax : matplotlib.axes._subplots.Axes3DSubplot
            The axis used for the plot.

        """
        if positions is None:
            positions = np.zeros((self._quat.shape[0], 3))
        positions = np.atleast_2d(positions).astype(np.float64)
        if positions.shape[0] != self._quat.shape[0]:
            raise ValueError("If provided, there must be the same number"
                             "of positions as orientations.")

        # Create view, up and right vectors from Rotation object
        views, ups, rights = self.as_view_up_right()

        kwargs.pop('color', None)

        ax = None
        if show_views:
            ax = haiopy.plot.quiver(
                positions, views, color=(1, 0, 0), **kwargs)
        if show_ups:
            ax = haiopy.plot.quiver(
                positions, ups, ax=ax, color=(0, 1, 0), **kwargs)
        if show_rights:
            ax = haiopy.plot.quiver(
                positions, rights, ax=ax, color=(0, 0, 1), **kwargs)

    def as_view_up_right(self):
        """Get Orientations as a view, up, and right vector.

        Orientations are internally stored as quaternions for better spherical
        linear interpolation (SLERP) and spherical harmonics operations.
        More intuitionally, they can be expressed as view and and up of vectors
        which cannot be collinear. In this case are restricted to be
        perpendicular to minimize rounding errors.

        Returns
        ----------
        vector_triple: ndarray, shape (N, 3), normalized vectors
            - views, see `Orientations.from_view_up.__doc__`
            - ups, see `Orientations.from_view_up.__doc__`
            - rights, see `Orientations.from_view_up.__doc__`
                A single vector or a stack of vectors, pointing to the right of
                the object, constructed as a cross product of ups and rights.
        """
        # Apply self as a Rotation (base class) on eye i.e. generate orientions
        # as rotations relative to standard basis in 3d
        vector_triple = super().as_matrix()
        if vector_triple.ndim == 3:
            return np.swapaxes(vector_triple, 0, 1)
        return vector_triple

    def __setitem__(self, idx, val):
        """
        Assign orientations(s) at given index(es) from object.

        Parameters
        ----------
        idx : see NumPy Indexing
        val : array_like quaternion(s), shape (N, 4) or (4,)
        """
        if isinstance(val, Orientations) and len(val) == 1:
            val = val[0].as_quat()
        quat = np.atleast_2d(val)
        if quat.ndim > 2 or quat.shape[-1] != 4:
            raise ValueError(f"Expected assigned value to have shape"
                             f" or (1, 4), got {quat.shape}")
        self._quat[idx] = quat

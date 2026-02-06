"""This module contains the Orientations class."""
from scipy.spatial.transform import Rotation
import numpy as np
import warnings

import pyfar as pf
if np.__version__ < '2.0.0':
    from numpy import VisibleDeprecationWarning
else:
    from numpy.exceptions import VisibleDeprecationWarning

# this warning needs to be caught and appears if numpy array are generated
# from nested lists containing lists of unequal lengths,
#  e.g., [[1, 0, 0], [1, 0]]
warnings.filterwarnings("error", category=VisibleDeprecationWarning)


class Orientations(Rotation):
    """
    This class for Orientations in the three-dimensional space,
    is a subclass of :py:class:`scipy:scipy.spatial.transform.Rotation` and
    equally based on quaternions of shape (N, 4). It inherits all methods of
    the Rotation class and adds the creation from perpendicular view and up
    vectors through :py:func:`~from_view_up` and a convenient plot function
    :py:func:`~show`.

    An orientation can be visualized with the triple of view, up and right
    vectors and it is tied to the object's local coordinate system.
    Alternatively the object's orientation can be illustrated with help of the
    right hand: Thumb (view), forefinger (up) and middle finger (right).

    Examples
    --------
    >>> from pyfar import Orientations
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

    def __init__(self, quat=None, normalize=True, copy=True, **kwargs):
        if quat is None:
            quat = np.array([0., 0., 0., 1.])
        super().__init__(quat, copy=copy, normalize=normalize, **kwargs)

    @classmethod
    def from_matrix(cls, matrix, assume_valid=False):
        """
        Initialize from rotation matrix.

        Rotations in 3 dimensions can be represented with 3 x 3 orthogonal
        matrices [#]_. If the input is not orthogonal, an approximation is
        created by orthogonalizing the input matrix using the method described
        in [#]_, and then converting the orthogonal rotation matrices to
        quaternions using the algorithm described in [#]_. Matrices must be
        right-handed.

        Parameters
        ----------
        matrix : array_like, shape (..., 3, 3)
            A single matrix or an ND array of matrices, where the last two
            dimensions contain the rotation matrices.
        assume_valid : bool, optional
            Must be False unless users can guarantee the input is a valid
            rotation matrix, i.e. it is orthogonal, rows and columns have unit
            norm and the determinant is 1. Setting this to True without
            ensuring these properties is unsafe and will silently lead to
            incorrect results. If True, normalization steps are skipped, which
            can improve runtime performance.

        References
        ----------
        .. [#] https://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions
        .. [#] https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
        .. [#] F. Landis Markley, "Unit Quaternion from Rotation Matrix",
               Journal of guidance, control, and dynamics vol. 31.2, pp.
               440-442, 2008.
        """
        # Methods inherited from scipy.spatial.transforms.Rotation return an
        # instance of the parent class for scipy >= 1.17 and are therefore
        # wrapped to return an Orientations-object instead.
        rot = Rotation.from_matrix(matrix=matrix, assume_valid=assume_valid)
        quat = rot.as_quat()
        return cls(quat)

    @classmethod
    def from_quat(cls, quat, scalar_first=False):
        """
        Initialize from quaternions.

        Rotations in 3 dimensions can be represented using unit norm
        quaternions [#]_.

        The 4 components of a quaternion are divided into a scalar part ``w``
        and a vector part ``(x, y, z)`` and can be expressed from the angle
        ``theta`` and the axis ``n`` of a rotation as follows::

            w = cos(theta / 2)
            x = sin(theta / 2) * n_x
            y = sin(theta / 2) * n_y
            z = sin(theta / 2) * n_z

        There are 2 conventions to order the components in a quaternion:

        - scalar-first order -- ``(w, x, y, z)``
        - scalar-last order -- ``(x, y, z, w)``

        The choice is controlled by `scalar_first` argument.
        By default, it is False and the scalar-last order is assumed.

        Advanced users may be interested in the "double cover" of 3D space by
        the quaternion representation [#]_. As of version 1.11.0, the
        following subset (and only this subset) of operations on a `Rotation`
        ``r`` corresponding to a quaternion ``q`` are guaranteed to preserve
        the double cover property: ``r = Rotation.from_quat(q)``,
        ``r.as_quat(canonical=False)``, ``r.inv()``, and composition using the
        ``*`` operator such as ``r*r``.

        Parameters
        ----------
        quat : array_like, shape (..., 4)
            Each row is a (possibly non-unit norm) quaternion representing an
            active rotation. Each quaternion will be normalized to unit norm.
        scalar_first : bool, optional
            Whether the scalar component goes first or last.
            Default is False, i.e. the scalar-last order is assumed.

        References
        ----------
        .. [#] https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
        .. [#] Hanson, Andrew J. "Visualizing quaternions."
            Morgan Kaufmann Publishers Inc., San Francisco, CA. 2006.
        """
        # Methods inherited from scipy.spatial.transforms.Rotation return an
        # instance of the parent class for scipy >= 1.17 and are therefore
        # wrapped to return an Orientations-object instead.
        rot = Rotation.from_quat(quat=quat, scalar_first=scalar_first)
        quat = rot.as_quat()
        return cls(quat)

    @classmethod
    def from_rotvec(cls, rotvec, degrees=False):
        """Initialize from rotation vectors.

        A rotation vector is a 3 dimensional vector which is co-directional to
        the axis of rotation and whose norm gives the angle of rotation [#]_.

        Parameters
        ----------
        rotvec : array_like, shape (..., 3)
            A single vector or an ND array of vectors, where the last dimension
            contains the rotation vectors.
        degrees : bool, optional
            If True, then the given magnitudes are assumed to be in degrees.
            Default is False.

        References
        ----------
        .. [#] https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation#Rotation_vector
        """
        # Methods inherited from scipy.spatial.transforms.Rotation return an
        # instance of the parent class for scipy >= 1.17 and are therefore
        # wrapped to return an Orientations-object instead.
        rot = Rotation.from_rotvec(rotvec=rotvec, degrees=degrees)
        quat = rot.as_quat()
        return cls(quat)

    @classmethod
    def from_mrp(cls, mrp):
        """
        Initialize from Modified Rodrigues Parameters (MRPs).

        MRPs are a 3 dimensional vector co-directional to the axis of rotation
        and whose magnitude is equal to ``tan(theta / 4)``, where ``theta`` is
        the angle of rotation (in radians) [#]_.

        MRPs have a singularity at 360 degrees which can be avoided by ensuring
        the angle of rotation does not exceed 180 degrees, i.e. switching the
        direction of the rotation when it is past 180 degrees.

        Parameters
        ----------
        mrp : array_like, shape (..., 3)
            A single vector or an ND array of vectors, where the last dimension
            contains the rotation parameters.

        References
        ----------
        .. [#] Shuster, M. D. "A Survey of Attitude Representations",
               The Journal of Astronautical Sciences, Vol. 41, No.4, 1993,
               pp. 475-476
        """
        # Methods inherited from scipy.spatial.transforms.Rotation return an
        # instance of the parent class for scipy >= 1.17 and are therefore
        # wrapped to return an Orientations-object instead.
        rot = Rotation.from_mrp(mrp)
        quat = rot.as_quat()
        return cls(quat)

    @classmethod
    def from_euler(cls, seq, angles, degrees=False):
        """Initialize from Euler angles.

        Rotations in 3-D can be represented by a sequence of 3
        rotations around a sequence of axes. In theory, any three axes spanning
        the 3-D Euclidean space are enough. In practice, the axes of rotation
        are chosen to be the basis vectors.

        The three rotations can either be in a global frame of reference
        (extrinsic) or in a body centred frame of reference (intrinsic), which
        is attached to, and moves with, the object under rotation [#]_.

        Parameters
        ----------
        seq : string
            Specifies sequence of axes for rotations. Up to 3 characters
            belonging to the set {'X', 'Y', 'Z'} for intrinsic rotations, or
            {'x', 'y', 'z'} for extrinsic rotations. Extrinsic and intrinsic
            rotations cannot be mixed in one function call.
        angles : float or array_like, shape (...,  [1 or 2 or 3])
            Euler angles specified in radians (`degrees` is False) or degrees
            (`degrees` is True).
            Each character in `seq` defines one axis around which `angles`
            turns. The resulting rotation has the shape
            np.atleast_1d(angles).shape[:-1]. Dimensionless angles are thus
            only valid for single character `seq`.

        degrees : bool, optional
            If True, then the given angles are assumed to be in degrees.
            Default is False.

        References
        ----------
        .. [#] https://en.wikipedia.org/wiki/Euler_angles#Definition_by_intrinsic_rotations
        """
        # Methods inherited from scipy.spatial.transforms.Rotation return an
        # instance of the parent class for scipy >= 1.17 and are therefore
        # wrapped to return an Orientations-object instead.
        rot = Rotation.from_euler(seq=seq, angles=angles, degrees=degrees)
        quat = rot.as_quat()
        return cls(quat)

    @classmethod
    def from_davenport(cls, axes, order, angles, degrees=False):
        """Initialize from Davenport angles.

        Rotations in 3-D can be represented by a sequence of 3
        rotations around a sequence of axes.

        The three rotations can either be in a global frame of reference
        (extrinsic) or in a body centred frame of reference (intrinsic), which
        is attached to, and moves with, the object under rotation [#]_.

        For both Euler angles and Davenport angles, consecutive axes must
        be are orthogonal (``axis2`` is orthogonal to both ``axis1`` and
        ``axis3``). For Euler angles, there is an additional relationship
        between ``axis1`` or ``axis3``, with two possibilities:

            - ``axis1`` and ``axis3`` are also orthogonal (asymmetric sequence)
            - ``axis1 == axis3`` (symmetric sequence)

        For Davenport angles, this last relationship is relaxed [#]_, and only
        the consecutive orthogonal axes requirement is maintained.

        Parameters
        ----------
        axes : array_like, shape (3,) or (..., [1 or 2 or 3], 3)
            Axis of rotation, if one dimensional. If two or more dimensional,
            describes the sequence of axes for rotations, where each
            axes[..., i, :] is the ith axis. If more than one axis is given,
            then the second axis must be orthogonal to both the first and third
            axes.
        order : string
            If it is equal to 'e' or 'extrinsic', the sequence will be
            extrinsic. If it is equal to 'i' or 'intrinsic', sequence
            will be treated as intrinsic.
        angles : float or array_like, shape (..., [1 or 2 or 3])
            Angles specified in radians (`degrees` is False) or degrees
            (`degrees` is True).
            Each angle i in the last dimension of `angles` turns around the
            corresponding axis axis[..., i, :]. The resulting rotation has the
            shape np.broadcast_shapes(np.atleast_2d(axes).shape[:-2],
            np.atleast_1d(angles).shape[:-1]) Dimensionless angles are thus
            only valid for a single axis.

        degrees : bool, optional
            If True, then the given angles are assumed to be in degrees.
            Default is False.

        References
        ----------
        .. [#] https://en.wikipedia.org/wiki/Euler_angles#Definition_by_intrinsic_rotations
        .. [#] Shuster, Malcolm & Markley, Landis. (2003). Generalization of
               the Euler Angles. Journal of the Astronautical Sciences. 51.
               123-132. 10.1007/BF03546304.
        """
        # Methods inherited from scipy.spatial.transforms.Rotation return an
        # instance of the parent class for scipy >= 1.17 and are therefore
        # wrapped to return an Orientations-object instead.
        rot = Rotation.from_davenport(axes=axes, order=order, angles=angles,
                                      degrees=degrees)
        quat = rot.as_quat()
        return cls(quat)

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
        views : array_like, shape (N, 3) or (3,), Coordinates
            A single vector or a stack of vectors, giving the look-direction of
            an object in three-dimensional space, e.g. from a listener, or the
            acoustic axis of a loudspeaker, or the direction of a main lobe.
            Views can also be passed as a Coordinates object.

        ups : array_like, shape (N, 3) or (3,), Coordinates
            A single vector or a stack of vectors, giving the up-direction of
            an object, which is usually the up-direction in world-space. Views
            can also be passed as a Coordinates object.

        Returns
        -------
        orientations : Orientations
            Object containing the orientations represented by quaternions.
        """

        # init views and up
        try:
            views = np.atleast_2d(views).astype(np.float64)
            ups = np.atleast_2d(ups).astype(np.float64)
        except VisibleDeprecationWarning as exc:
            raise ValueError(
                "Expected `views` and `ups` to have shape (N, 3)") from exc

        # check views and ups
        if (views.ndim > 2 or views.shape[-1] != 3 or
                ups.ndim > 2 or ups.shape[-1] != 3):
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

        if not (np.all(np.linalg.norm(views, axis=1)) and
                np.all(np.linalg.norm(ups, axis=1))):
            raise ValueError("View and Up Vectors must have a length.")
        if not np.allclose(0, np.einsum('ij,kj->k', views, ups)):
            raise ValueError("View and Up vectors must be perpendicular.")

        # Assuming that the direction of the cross product is defined
        # by the right-hand rule
        rights = np.cross(views, ups)

        rotation_matrix = np.asarray([views, ups, rights])
        rotation_matrix = np.swapaxes(rotation_matrix, 0, 1)

        return cls.from_matrix(rotation_matrix)

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
        kwargs : dict
            Additional arguments passed to :py:func:`pyfar.plot.quiver`.

        Returns
        -------
        ax : :py:class:`~mpl_toolkits.mplot3d.axes3d.Axes3D`
            The axis used for the plot.

        """
        if positions is None:
            positions = np.zeros((self.as_quat().shape[0], 3))
        positions = np.atleast_2d(positions).astype(np.float64)
        if positions.shape[0] != self.as_quat().shape[0]:
            raise ValueError("If provided, there must be the same number"
                             "of positions as orientations.")

        # Create view, up and right vectors from Rotation object
        views, ups, rights = self.as_view_up_right()

        kwargs.pop('color', None)

        ax = None
        if show_views:
            ax = pf.plot.quiver(
                positions, views, color=pf.plot.color('r'), **kwargs)
        if show_ups:
            ax = pf.plot.quiver(
                positions, ups, ax=ax, color=pf.plot.color('g'), **kwargs)
        if show_rights:
            ax = pf.plot.quiver(
                positions, rights, ax=ax, color=pf.plot.color('b'), **kwargs)

    def as_view_up_right(self):
        """Get Orientations as a view, up, and right vector.

        Orientations are internally stored as quaternions for better spherical
        linear interpolation (SLERP) and spherical harmonics operations.
        More intuitionally, they can be expressed as view and and up of vectors
        which cannot be collinear. In this case are restricted to be
        perpendicular to minimize rounding errors.

        Returns
        -------
        vector_triple: ndarray, shape (N, 3), normalized vectors
            - views, see :py:func:`Orientations.from_view_up`
            - ups, see :py:func:`Orientations.from_view_up`
            - rights, see :py:func:`Orientations.from_view_up`
                A single vector or a stack of vectors, pointing to the right of
                the object, constructed as a cross product of ups and rights.
        """
        # Apply self as a Rotation (base class) on eye i.e. generate orientions
        # as rotations relative to standard basis in 3d
        vector_triple = super().as_matrix()
        if vector_triple.ndim == 3:
            return np.swapaxes(vector_triple, 0, 1)
        return vector_triple

    def copy(self):
        """Return a deep copy of the Orientations object."""
        return self.from_quat(self.as_quat())

    def _encode(self):
        """Return object in a proper encoding format."""
        # Use public interface of the scipy super-class to prevent
        # error in case of chaning super-class implementations
        return {'quat': self.as_quat()}

    @classmethod
    def _decode(cls, obj_dict):
        """Decode object based on its respective `_encode` counterpart."""
        return cls.from_quat(obj_dict['quat'])

    def __setitem__(self, idx, val):
        """
        Assign orientations(s) at given index(es) from object.

        Parameters
        ----------
        idx : indexes
            see NumPy Indexing
        val : array_like
            quaternion(s), shape (N, 4) or (4,)
        """
        if isinstance(val, (Rotation, Orientations)):
            quat = np.atleast_2d(val.as_quat())
        else:
            quat = np.atleast_2d(val)

        if quat.ndim > 2 or quat.shape[-1] != 4:
            raise ValueError(f"Expected assigned value to have shape"
                             f" or (1, 4), got {quat.shape}")

        quats = self.as_quat()
        quats[idx] = quat
        self = Orientations.from_quat(quats)

    def __getitem__(self, idx):
        """Get orientation(s) at given indices.

        Parameters
        ----------
        idx : indexes
            see NumPy Indexing
        """
        # Methods inherited from scipy.spatial.transforms.Rotation return an
        # instance of the parent class for scipy >= 1.17 and are therefore
        # wrapped to return an Orientations-object instead.
        quat = self.as_quat()
        return Orientations(quat[idx])

    def __mul__(self, other):
        """
        Multiply Orientations object with another Orientations or
        Rotation object.

        Parameters
        ----------
        other : Orientations or Rotation
            The object to multiply with. If an Orientations object is provided,
            it will be converted to a Rotation object for the multiplication.
        """
        # Methods inherited from scipy.spatial.transforms.Rotation return an
        # instance of the parent class for scipy >= 1.17 and are therefore
        # wrapped to return an Orientations-object instead.
        if isinstance(other, Orientations):
            other = Rotation.from_quat(other.as_quat())
        if isinstance(other, Rotation):
            rot = Rotation.from_quat(self.as_quat()) * other
            return Orientations(rot.as_quat())

    def __rmul__(self, other):
        """
        Right multiplication of Orientations or Rotation with Orientations.

        Parameters
        ----------
        other : Orientations or Rotation
            The object to multiply with this Orientations object on the left
            side. If an Orientations object is provided, it will be converted
            to a Rotation object using quaternion representation.
        """
        # Methods inherited from scipy.spatial.transforms.Rotation return an
        # instance of the parent class for scipy >= 1.17 and are therefore
        # wrapped to return an Orientations-object instead.
        if isinstance(other, Orientations):
            other = Rotation.from_quat(other.as_quat())
        if isinstance(other, Rotation):
            rot = other * Rotation.from_quat(self.as_quat())
            return Orientations(rot.as_quat())

    def __eq__(self, other):
        """Check for equality of two objects."""
        return np.array_equal(self.as_quat(), other.as_quat())

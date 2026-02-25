"""This module contains the Rotation class."""
from scipy.spatial.transform import Rotation as scRotation
import numpy as np
import warnings

if np.__version__ < '2.0.0':
    from numpy import VisibleDeprecationWarning
else:
    from numpy.exceptions import VisibleDeprecationWarning

# this warning needs to be caught and appears if numpy array are generated
# from nested lists containing lists of unequal lengths,
#  e.g., [[1, 0, 0], [1, 0]]
warnings.filterwarnings("error", category=VisibleDeprecationWarning)


class Rotation():
    """
    Rotation in the three-dimensional space.

    This class is largely based on '
    ':py:class:`scipy:scipy.spatial.transform.Rotation and wraps all '
    'functionality that scipy's Rotation class provides.'
    In addition the pyfar Rotation class adds the creation from perpendicular
    view and up vectors through :py:func:`~from_view_up`, and the
    representation as view / up in :py:func:`~as_view_up`.

    A rotation can be visualized with the triple of view, up and right
    vectors and it is tied to the object's local coordinate system.
    Alternatively the object's rotation can be illustrated with help of the
    right hand: Thumb (view), forefinger (up) and middle finger (right).

    Examples
    --------
    >>> from pyfar import Rotation
    >>> views = [[1, 0, 0], [2, 0, 0]]
    >>> ups = [[0, 1, 0], [0, -2, 0]]
    >>> rotations = Rotation.from_view_up(views, ups)

    Visualize rotations at certain positions:

    >>> positions = [[0, 0.5, 0], [0, -0.5, 0]]
    >>> rotations.show(positions)

    Rotate by 45 degree in x-direction:

    >>> rot_x45 = Rotation.from_euler('x', 45, degrees=True)
    >>> rotations = rotations * rot_x45
    >>> rotations.show(positions)

    To create `Rotation` objects use ``from_...`` methods.
    """

    def __init__(self):
        raise RuntimeError("Rotation objects must be created using one of "
                           "the `from_...` methods")

    def __repr__(self):
        """String representation of Rotation object."""
        repr_string = \
              f"Pyfar.Rotation with {self.cshape} rotations."
        return repr_string

    def __iter__(self):
        """Iterate over rotations."""
        if self.as_quat().ndim == 1:
            raise TypeError("Single rotation is not iterable.")
        quat = self.as_quat().reshape((-1, 4))
        for i in range(self.csize):
            yield self.from_quat(quat[i])

    def __getitem__(self, idx):
        """
        Get rotation(s) at given indices.

        Parameters
        ----------
        idx : indexes
            see NumPy Indexing
        """
        return self._from_scipy_rotation(self._rot[idx])

    def __setitem__(self, *args):
        """
        Assign rotation(s) at given index(es) from object.

        This is disabled for `Rotation`.
        """
        raise NotImplementedError(
            'Setting an item is disabled for pyfar '
            'Rotations. If you want to modify the Rotation, use an array '
            'representation like `as_quat()` or `as_matrix()` and create a '
            'new object.')

    def __mul__(self, other):
        """
        Multiply Rotation object with another Rotation or
        :py:class:`scipy:scipy.spatial.transform.Rotation`.

        Parameters
        ----------
        other : Rotation or :py:class:`scipy:scipy.spatial.transform.Rotation`
            The object to multiply with.
        """
        if isinstance(other, Rotation):
            other = scRotation.from_quat(other.as_quat())
        self._rot *= other
        return self

    def __rmul__(self, other):
        """
        Right multiplication of Rotation object with another Rotation or a
        :py:class:`scipy:scipy.spatial.transform.Rotation`.

        Parameters
        ----------
        other : Rotation or :py:class:`scipy:scipy.spatial.transform.Rotation`.
            The object to multiply with.
        """
        if isinstance(other, Rotation):
            other = scRotation.from_quat(other.as_quat())
        if isinstance(other, scRotation):
            self._rot *= other
            return self

    def __pow__(self, n):
        """Compose rotation with itself n times."""
        rot = self._rot.__pow__(n)
        return self._from_scipy_rotation(rot)

    def __eq__(self, other):
        """Check for equality of two objects."""
        return np.array_equal(self.as_quat(), other.as_quat())

    # properties
    @property
    def cshape(self):
        """Cshape of rotations."""
        quat = self._rot.as_quat()
        if quat.ndim==1:
            return (1,)
        return quat.shape[:-1]

    @property
    def csize(self):
        """Number of rotations."""
        return np.prod(self.cshape)


    # from-... methods
    @classmethod
    def from_davenport(cls, axes, order, angles, degrees=False):
        """
        Initialize from Davenport angles.

        Wraps :py:meth:`scipy:scipy.spatial.transform.Rotation.from_davenport`.

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

        Returns
        -------
        rotations : Rotation
            Object containing the rotations.

        References
        ----------
        .. [#] https://en.wikipedia.org/wiki/Euler_angles#Definition_by_intrinsic_rotations
        .. [#] Shuster, Malcolm & Markley, Landis. (2003). Generalization of
               the Euler Angles. Journal of the Astronautical Sciences. 51.
               123-132. 10.1007/BF03546304.
        """
        rot = scRotation.from_davenport(
            axes, order, angles, degrees=degrees)
        return cls._from_scipy_rotation(rot)

    @classmethod
    def from_euler(cls, seq, angles, degrees=False):
        """
        Initialize from Euler angles.

        Wraps :py:meth:`scipy:scipy.spatial.transform.Rotation.from_euler`.

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

        Returns
        -------
        rotations : Rotation
            Object containing the rotations.

        References
        ----------
        .. [#] https://en.wikipedia.org/wiki/Euler_angles#Definition_by_intrinsic_rotations
        """
        rot = scRotation.from_euler(
            seq, angles, degrees=degrees)
        return cls._from_scipy_rotation(rot)

    @classmethod
    def from_matrix(cls, matrix):
        """
        Initialize from rotation matrix.

        Wraps :py:meth:`scipy:scipy.spatial.transform.Rotation.from_matrix`.

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

        Returns
        -------
        rotations : Rotation
            Object containing the rotations.

        References
        ----------
        .. [#] https://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions
        .. [#] https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
        .. [#] F. Landis Markley, "Unit Quaternion from Rotation Matrix",
               Journal of guidance, control, and dynamics vol. 31.2, pp.
               440-442, 2008.
        """
        rot = scRotation.from_matrix(matrix)
        return cls._from_scipy_rotation(rot)

    @classmethod
    def from_mrp(cls, mrp):
        """
        Initialize from Modified Rodrigues Parameters (MRPs).

        Wraps :py:meth:`scipy:scipy.spatial.transform.Rotation.from_mrp`.

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

        Returns
        -------
        rotations : Rotation
            Object containing the rotations.

        References
        ----------
        .. [#] Shuster, M. D. "A Survey of Attitude Representations",
               The Journal of Astronautical Sciences, Vol. 41, No.4, 1993,
               pp. 475-476
        """
        rot = scRotation.from_mrp(mrp)
        return cls._from_scipy_rotation(rot)

    @classmethod
    def from_quat(cls, quat):
        """
        Initialize from quaternions.

        Wraps :py:meth:`scipy:scipy.spatial.transform.Rotation.from_quat`.

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

        Returns
        -------
        rotations : Rotation
            Object containing the rotations.

        References
        ----------
        .. [#] https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
        .. [#] Hanson, Andrew J. "Visualizing quaternions."
            Morgan Kaufmann Publishers Inc., San Francisco, CA. 2006.
        """
        rot = scRotation.from_quat(quat)
        return cls._from_scipy_rotation(rot)

    @classmethod
    def from_rotvec(cls, rotvec, degrees=False):
        """
        Initialize from rotation vectors.

        Wraps :py:meth:`scipy:scipy.spatial.transform.Rotation.from_rotvec`.

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

        Returns
        -------
        rotations : Rotation
            Object containing the rotations.

        References
        ----------
        .. [#] https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation#Rotation_vector
        """
        rot = scRotation.from_rotvec(rotvec, degrees)
        return cls._from_scipy_rotation(rot)

    @classmethod
    def from_view_up(cls, views, ups):
        """
        Initialize Rotation from a view an up vector.

        Rotations are internally stored as quaternions for better spherical
        linear interpolation (SLERP) and spherical harmonics operations.
        More intuitionally, they can be expressed as view and up vectors
        which cannot be collinear. In this case, they are restricted to be
        perpendicular to minimize rounding errors.

        Parameters
        ----------
        views : array_like, shape (..., 3) or (3,), Coordinates
            A single vector or a stack of vectors, giving the look-direction of
            an object in three-dimensional space, e.g. from a listener, or the
            acoustic axis of a loudspeaker, or the direction of a main lobe.
            Views can also be passed as a Coordinates object.

        ups : array_like, shape (..., 3) or (3,), Coordinates
            A single vector or a stack of vectors, giving the up-direction of
            an object, which is usually the up-direction in world-space. Views
            can also be passed as a Coordinates object.

        Returns
        -------
        rotations : Rotation
            Object containing the rotations.
        """
        # init views and up
        views = np.atleast_2d(views)
        ups = np.atleast_2d(ups)

        # determine broadcasted shape
        try:
            shape = np.broadcast_shapes(views.shape, ups.shape)
        except ValueError as exc:
            message = (
                f"shape missmatch: `views` {views.shape} and `ups` {ups.shape}"
                " cannot be broadcasted to a single shape"
            )
            raise ValueError(message) from exc

        # check shape
        if shape[-1] != 3:
            raise ValueError(f"Expected `views` and `ups` to have shape "
                             f"(..., 3) or (3,), got {views.shape} and "
                             f"{ups.shape}")

        views = np.broadcast_to(views, shape)
        ups = np.broadcast_to(ups, shape)

        if not (np.all(np.linalg.norm(views, axis=-1)) and
                np.all(np.linalg.norm(ups, axis=-1))):
            raise ValueError("View and Up Vectors must have a length.")
        if not np.allclose(0, np.einsum('...j,...j->...', views, ups)):
            raise ValueError("View and Up vectors must be perpendicular.")

        # Assuming that the direction of the cross product is defined
        # by the right-hand rule
        rights = np.cross(views, ups)

        # In a standard Cartesian right-handed coordinate system,
        # these vectors are defined as [x, y, z] = [view, left, up], where
        # left is the same vector as -rights
        rotation_matrix = np.stack((views, -rights, ups), axis=-2)
        return cls.from_matrix(rotation_matrix)

    # other class methods
    @classmethod
    def align_vectors(cls, a, b, weights=None, return_sensitivity=False):
        r"""
        Estimate a rotation to optimally align two sets of vectors.

        Wraps :py:meth:`scipy:scipy.spatial.transform.Rotation.align_vectors`.

        Find a rotation between frames A and B which best aligns a set of
        vectors `a` and `b` observed in these frames. The following loss
        function is minimized to solve for the rotation matrix
        :math:`C`:

        .. math::

            L(C) = \\frac{1}{2} \\sum_{i = 1}^{n} w_i \\lVert \\mathbf{a}_i -
            C \\mathbf{b}_i \\rVert^2 ,

        where :math:`w_i`'s are the `weights` corresponding to each vector.

        The rotation is estimated with Kabsch algorithm [#]_, and solves what
        is known as the "pointing problem", or "Wahba's problem" [#]_.

        Note that the length of each vector in this formulation acts as an
        implicit weight. So for use cases where all vectors need to be
        weighted equally, you should normalize them to unit length prior to
        calling this method.

        There are two special cases. The first is if a single vector is given
        for `a` and `b`, in which the shortest distance rotation that aligns
        `b` to `a` is returned.

        The second is when one of the weights is infinity. In this case, the
        shortest distance rotation between the primary infinite weight vectors
        is calculated as above. Then, the rotation about the aligned primary
        vectors is calculated such that the secondary vectors are optimally
        aligned per the above loss function. The result is the composition
        of these two rotations. The result via this process is the same as the
        Kabsch algorithm as the corresponding weight approaches infinity in
        the limit. For a single secondary vector this is known as the
        "align-constrain" algorithm [#]_.

        For both special cases (single vectors or an infinite weight), the
        sensitivity matrix does not have physical meaning and an error will be
        raised if it is requested. For an infinite weight, the primary vectors
        act as a constraint with perfect alignment, so their contribution to
        `rssd` will be forced to 0 even if they are of different lengths.

        Parameters
        ----------
        a : array_like, shape (3,) or (N, 3)
            Vector components observed in initial frame A. Each row of `a`
            denotes a vector.
        b : array_like, shape (3,) or (N, 3)
            Vector components observed in another frame B. Each row of `b`
            denotes a vector.
        weights : array_like shape (N,), optional
            Weights describing the relative importance of the vector
            observations. If None (default), then all values in `weights` are
            assumed to be 1. One and only one weight may be infinity, and
            weights must be positive.
        return_sensitivity : bool, optional
            Whether to return the sensitivity matrix. See Notes for details.
            Default is False.

        Returns
        -------
        rotation : Rotation
            Best estimate of the rotation that transforms `b` to `a`.
        rssd : float
            Stands for "root sum squared distance". Square root of the weighted
            sum of the squared distances between the given sets of vectors
            after alignment. It is equal to ``sqrt(2 * minimum_loss)``, where
            ``minimum_loss`` is the loss function evaluated for the found
            optimal rotation.
            Note that the result will also be weighted by the vectors'
            magnitudes, so perfectly aligned vector pairs will have nonzero
            `rssd` if they are not of the same length. This can be avoided by
            normalizing them to unit length prior to calling this method,
            though note that doing this will change the resulting rotation.
        sensitivity_matrix : ndarray, shape (3, 3)
            Sensitivity matrix of the estimated rotation estimate as explained
            in Notes. Returned only when `return_sensitivity` is True. Not
            valid if aligning a single pair of vectors or if there is an
            infinite weight, in which cases an error will be raised.

        Notes
        -----
        The sensitivity matrix gives the sensitivity of the estimated rotation
        to small perturbations of the vector measurements. Specifically we
        consider the rotation estimate error as a small rotation vector of
        frame A. The sensitivity matrix is proportional to the covariance of
        this rotation vector assuming that the vectors in `a` was measured with
        errors significantly less than their lengths. To get the true
        covariance matrix, the returned sensitivity matrix must be multiplied
        by harmonic mean [#]_ of variance in each observation. Note that
        `weights` are supposed to be inversely proportional to the observation
        variances to get consistent results. For example, if all vectors are
        measured with the same accuracy of 0.01 (`weights` must be all equal),
        then you should multiple the sensitivity matrix by 0.01**2 to get the
        covariance.

        Refer to [#]_ for more rigorous discussion of the covariance
        estimation. See [#]_ for more discussion of the pointing problem and
        minimal proper pointing.

        This function does not support broadcasting or ND arrays with N > 2.

        References
        ----------
        .. [#] https://en.wikipedia.org/wiki/Kabsch_algorithm
        .. [#] https://en.wikipedia.org/wiki/Wahba%27s_problem
        .. [#] Magner, Robert,
                "Extending target tracking capabilities through trajectory and
                momentum setpoint optimization." Small Satellite Conference,
                2018.
        .. [#] https://en.wikipedia.org/wiki/Harmonic_mean
        .. [#] F. Landis Markley,
                "Attitude determination using vector observations: a fast
                optimal matrix algorithm", Journal of Astronautical Sciences,
                Vol. 41, No.2, 1993, pp. 261-280.
        .. [#] Bar-Itzhack, Itzhack Y., Daniel Hershkowitz, and Leiba Rodman,
               "Pointing in Real Euclidean Space", Journal of Guidance,
               Control, and Dynamics, Vol. 20, No. 5, 1997, pp. 916-922.
        """
        result = scRotation.align_vectors(a, b, weights, return_sensitivity)

        return (cls._from_scipy_rotation(result[0]), *result[1:])

    @classmethod
    def concatenate(cls, rotations):
        """
        Concatenate a sequence of Rotation objects into a single object.

        Wraps :py:meth:`scipy:scipy.spatial.transform.Rotation.concatenate`.

        This is useful if you want to, for example, take the mean of a set of
        rotations and need to pack them into a single object to do so.

        Parameters
        ----------
        rotations : sequence of Rotation objects
            The rotation to concatenate. If a single Rotation object is
            passed in, a copy is returned.

        Returns
        -------
        concatenated : Rotation
            The concatenated rotations.
        """
        if np.asarray(rotations).shape[0] == 1:
            return rotations[0].copy()

        rotations = [rotation._rot for rotation in rotations]

        rot = scRotation.concatenate(rotations)
        return cls._from_scipy_rotation(rot)

    @classmethod
    def identity(cls, num=None, *, shape=None):
        """
        Get identity rotation(s).

        Wraps :py:meth:`scipy:scipy.spatial.transform.Rotation.identity`.

        Composition with the identity rotation has no effect.

        Parameters
        ----------
        num : int or None, optional
            Number of identity rotations to generate. If None (default),
            then a single rotation is generated.
        shape : int or tuple of ints, optional
            Shape of identity rotations to generate. If specified, `num`
            must be None.

        Returns
        -------
        identity : Rotation
            The identity rotation.
        """
        rot  = scRotation.identity(num, shape=shape)
        return cls._from_scipy_rotation(rot)

    @classmethod
    def random(cls, num=None, rng=None, *, shape=None):
        """
        Generate rotations that are uniformly distributed on a sphere.

        Wraps :py:meth:`scipy:scipy.spatial.transform.Rotation.random`.

        Formally, the rotations follow the Haar-uniform distribution over
        the SO(3) group.

        Parameters
        ----------
        num : int or None, optional
            Number of random rotations to generate. If None (default), then
            a single rotation is generated.
        rng : `numpy.random.Generator`, optional
            Pseudorandom number generator state. When `rng` is None, a new
            `numpy.random.Generator` is created using entropy from the
            operating system. Types other than `numpy.random.Generator` are
            passed to `numpy.random.default_rng` to instantiate a `Generator`.
        shape : tuple of ints, optional
            Shape of random rotations to generate. If specified, `num` must
            be None.

        Returns
        -------
        random_rotation : Rotation
            Contains a single rotation if `num` is None. Otherwise contains
            a stack of `num` rotation.

        Notes
        -----
        This function is optimized for efficiently sampling random rotation
        matrices in three dimensions. For generating random rotation matrices
        in higher dimensions, see `scipy.stats.special_ortho_group`.
        """
        rot = scRotation.random(num, rng, shape=shape)
        return cls._from_scipy_rotation(rot)

    # private class methods
    @classmethod
    def _decode(cls, obj_dict):
        """Decode object based on its respective `_encode` counterpart."""
        return cls.from_quat(obj_dict['quat'])

    @classmethod
    def _from_scipy_rotation(cls, sc_rotation):
        """Internal helper to create instance from scipy Rotation."""
        instance = cls.__new__(cls)
        instance._rot = sc_rotation
        return instance

    # Instance methods
    def as_davenport(self, axes, order, degrees=False,
                     suppress_warnings=False):
        """
        Represent as Davenport angles.

        Wraps :py:meth:`scipy:scipy.spatial.transform.Rotation.as_davenport`.


        Any rotation can be expressed as a composition of 3 elementary
        rotations.

        For both Euler angles and Davenport angles, consecutive axes must
        be are orthogonal (``axis2`` is orthogonal to both ``axis1`` and
        ``axis3``). For Euler angles, there is an additional relationship
        between ``axis1`` or ``axis3``, with two possibilities:

            - ``axis1`` and ``axis3`` are also orthogonal (asymmetric sequence)
            - ``axis1 == axis3`` (symmetric sequence)

        For Davenport angles, this last relationship is relaxed [#]_, and only
        the consecutive orthogonal axes requirement is maintained.

        A slightly modified version of the algorithm from [#]_ has been used to
        calculate Davenport angles for the rotation about a given sequence of
        axes.

        Davenport angles, just like Euler angles, suffer from the problem of
        gimbal lock [#]_, where the representation loses a degree of freedom
        and it is not possible to determine the first and third angles
        uniquely. In this case, a warning is raised (unless the
        ``suppress_warnings`` option is used), and the third angle is set
        to zero. Note however that the returned angles still represent the
        correct rotation.

        Parameters
        ----------
        axes : array_like, shape (..., [1 or 2 or 3], 3) or (..., 3)
            Axis of rotation, if one dimensional. If N dimensional, describes
            the sequence of axes for rotations, where each axes[..., i, :] is
            the ith axis. If more than one axis is given, then the second axis
            must be orthogonal to both the first and third axes.
        order : string
            If it belongs to the set {'e', 'extrinsic'}, the sequence will be
            extrinsic. If it belongs to the set {'i', 'intrinsic'}, sequence
            will be treated as intrinsic.
        degrees : boolean, optional
            Returned angles are in degrees if this flag is True, else they are
            in radians. Default is False.
        suppress_warnings : boolean, optional
            Disable warnings about gimbal lock. Default is False.

        Returns
        -------
        angles : ndarray, shape (..., 3)
            Shape depends on shape of inputs used to initialize object.
            The returned angles are in the range:

            - First angle belongs to [-180, 180] degrees (both inclusive)
            - Third angle belongs to [-180, 180] degrees (both inclusive)
            - Second angle belongs to a set of size 180 degrees,
              given by: ``[-abs(lambda), 180 - abs(lambda)]``, where ``lambda``
              is the angle between the first and third axes.

        References
        ----------
        .. [#] Shuster, Malcolm & Markley, Landis. (2003). Generalization of
               the Euler Angles. Journal of the Astronautical Sciences. 51.
               123-132. 10.1007/BF03546304.
        .. [#] Bernardes E, Viollet S (2022) Quaternion to Euler angles
               conversion: A direct, general and computationally efficient
               method. PLoS ONE 17(11): e0276302. 10.1371/journal.pone.0276302
        .. [#] https://en.wikipedia.org/wiki/Gimbal_lock#In_applied_mathematics
        """

        return self._rot.as_davenport(axes, order, degrees, suppress_warnings)

    def as_euler(self, seq, degrees=False, suppress_warnings=False):
        """
        Represent as Euler angles.

        Wraps :py:meth:`scipy:scipy.spatial.transform.Rotation.as_euler`.

        Any rotation can be expressed as a composition of 3 elementary
        rotations. Once the axis sequence has been chosen, Euler angles define
        the angle of rotation around each respective axis [#]_.

        The algorithm from [#]_ has been used to calculate Euler angles for the
        rotation about a given sequence of axes.

        Euler angles suffer from the problem of gimbal lock [#]_, where the
        representation loses a degree of freedom and it is not possible to
        determine the first and third angles uniquely. In this case,
        a warning is raised (unless the ``suppress_warnings`` option is used),
        and the third angle is set to zero. Note however that the returned
        angles still represent the correct rotation.

        Parameters
        ----------
        seq : string, length 3
            3 characters belonging to the set {'X', 'Y', 'Z'} for intrinsic
            rotations, or {'x', 'y', 'z'} for extrinsic rotations [#]_.
            Adjacent axes cannot be the same.
            Extrinsic and intrinsic rotations cannot be mixed in one function
            call.
        degrees : boolean, optional
            Returned angles are in degrees if this flag is True, else they are
            in radians. Default is False.
        suppress_warnings : boolean, optional
            Disable warnings about gimbal lock. Default is False.

        Returns
        -------
        angles : ndarray, shape (..., 3)
            Shape depends on shape of inputs used to initialize object.
            The returned angles are in the range:

            - First angle belongs to [-180, 180] degrees (both inclusive)
            - Third angle belongs to [-180, 180] degrees (both inclusive)
            - Second angle belongs to:

                - [-90, 90] degrees if all axes are different (like xyz)
                - [0, 180] degrees if first and third axes are the same
                  (like zxz)

        References
        ----------
        .. [#] https://en.wikipedia.org/wiki/Euler_angles#Definition_by_intrinsic_rotations
        .. [#] Bernardes E, Viollet S (2022) Quaternion to Euler angles
               conversion: A direct, general and computationally efficient
               method. PLoS ONE 17(11): e0276302.
               https://doi.org/10.1371/journal.pone.0276302
        .. [#] https://en.wikipedia.org/wiki/Gimbal_lock#In_applied_mathematics
        """

        return self._rot.as_euler(seq, degrees,
                                  suppress_warnings=suppress_warnings)

    def as_matrix(self):
        """
        Represent as rotation matrix.

        Wraps :py:meth:`scipy:scipy.spatial.transform.Rotation.as_matrix`.

        3D rotations can be represented using rotation matrices, which
        are 3 x 3 real orthogonal matrices with determinant equal to +1 [#]_.

        Returns
        -------
        matrix : ndarray, shape (..., 3)
            Shape depends on shape of inputs used for initialization.

        References
        ----------
        .. [#] https://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions
        """

        return self._rot.as_matrix()

    def as_mrp(self):
        """
        Represent as Modified Rodrigues Parameters (MRPs).

        Wraps :py:meth:`scipy:scipy.spatial.transform.Rotation.as_mrp`.

        MRPs are a 3 dimensional vector co-directional to the axis of rotation
        and whose magnitude is equal to ``tan(theta / 4)``, where ``theta`` is
        the angle of rotation (in radians) [#]_.

        MRPs have a singularity at 360 degrees which can be avoided by ensuring
        the angle of rotation does not exceed 180 degrees, i.e. switching the
        direction of the rotation when it is past 180 degrees. This function
        will always return MRPs corresponding to a rotation of less than or
        equal to 180 degrees.

        Returns
        -------
        mrps : ndarray, shape (..., 3)
            Shape depends on shape of inputs used for initialization.

        References
        ----------
        .. [#] Shuster, M. D. "A Survey of Attitude Representations",
               The Journal of Astronautical Sciences, Vol. 41, No.4, 1993,
               pp. 475-476
        """

        return self._rot.as_mrp()

    def as_quat(self, canonical=False, *, scalar_first=False):
        """Represent as quaternions.

        Wraps :py:meth:`scipy:scipy.spatial.transform.Rotation.as_quat`.

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
        By default, it is False and the scalar-last order is used.

        The mapping from quaternions to rotations is
        two-to-one, i.e. quaternions ``q`` and ``-q``, where ``-q`` simply
        reverses the sign of each component, represent the same spatial
        rotation.

        Parameters
        ----------
        canonical : `bool`, default False
            Whether to map the redundant double cover of rotation space to a
            unique "canonical" single cover. If True, then the quaternion is
            chosen from {q, -q} such that the w term is positive. If the w term
            is 0, then the quaternion is chosen such that the first nonzero
            term of the x, y, and z terms is positive.
        scalar_first : bool, optional
            Whether the scalar component goes first or last.
            Default is False, i.e. the scalar-last order is used.

        Returns
        -------
        quat : `numpy.ndarray`, shape (..., 4)
            Shape depends on shape of inputs used for initialization.

        References
        ----------
        .. [#] https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
        """
        return self._rot.as_quat(canonical, scalar_first=scalar_first)

    def as_rotvec(self, degrees=False):
        """
        Represent as rotation vectors.

        Wraps :py:meth:`scipy:scipy.spatial.transform.Rotation.as_rotvec`.

        A rotation vector is a 3 dimensional vector which is co-directional to
        the axis of rotation and whose norm gives the angle of rotation [#]_.

        Parameters
        ----------
        degrees : boolean, optional
            Returned magnitudes are in degrees if this flag is True, else they
            are in radians. Default is False.

        Returns
        -------
        rotvec : ndarray, shape (..., 3)
            Shape depends on shape of inputs used for initialization.

        References
        ----------
        .. [#] https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation#Rotation_vector
        """

        return self._rot.as_rotvec(degrees)

    def as_view_up(self):
        """Get Rotation as a view, up, and right vector.

        Rotation are internally stored as quaternions for better spherical
        linear interpolation (SLERP) and spherical harmonics operations.
        More intuitionally, they can be expressed as view and and up of vectors
        which cannot be collinear. In this case are restricted to be
        perpendicular to minimize rounding errors.

        Returns
        -------
        vector_triple: ndarray, shape (..., 3), normalized vectors
            - views, see :py:func:`Rotation.from_view_up`
            - ups, see :py:func:`Rotation.from_view_up`
        """
        vector_triple = self.as_matrix()
        views, _, ups = np.split(vector_triple, 3, axis=-2)

        views = np.squeeze(views, axis=-2)
        ups   = np.squeeze(ups, axis=-2)

        return views, ups

    def copy(self):
        """Return a deep copy of the Rotation object."""
        return self.from_quat(self.as_quat())

    def inv(self):
        """
        Invert this rotation.

        Wraps :py:meth:`scipy:scipy.spatial.transform.Rotation.inv`.

        Composition of an rotation with its inverse results in an identity
        transformation.

        Returns
        -------
        inverse : Rotation
            Object containing inverse of the rotations in the current
            instance.
        """
        self._rot = self._rot.inv()
        return self

    def mean(self, weights=None, axis=None):
        r"""
        Get the mean of the rotations.

        Wraps :py:meth:`scipy:scipy.spatial.transform.Rotation.mean`.

        The mean used is the chordal L2 mean (also called the projected or
        induced arithmetic mean) [#]_. If ``A`` is a set of rotation matrices,
        then the mean ``M`` is the rotation matrix that minimizes the
        following loss function:

        .. math::

            L(M) = \\sum_{i = 1}^{n} w_i \\lVert \\mathbf{A}_i -
            \\mathbf{M} \\rVert^2 ,

        where :math:`w_i`'s are the `weights` corresponding to each matrix.

        Parameters
        ----------
        weights : array_like shape (..., N), optional
            Weights describing the relative importance of the rotations. If
            None (default), then all values in `weights` are assumed to be
            equal. If given, the shape of `weights` must be broadcastable to
            the rotation shape. Weights must be non-negative.
        axis : None, int, or tuple of ints, optional
            Axis or axes along which the means are computed. The default is to
            compute the mean of all rotations.

        Returns
        -------
        mean : Rotation
            Single rotation containing the mean of the rotations in the
            current instance.

        References
        ----------
        .. [#] Hartley, Richard, et al.,
                "Rotation Averaging", International Journal of Computer Vision
                103, 2013, pp. 267-305.
        """
        return self._from_scipy_rotation(self._rot.mean(weights, axis))

    def reduce(self, left=None, right=None, return_indices=False):
        """Reduce this rotation with the provided rotation groups.

        Wraps :py:meth:`scipy:scipy.spatial.transform.Rotation.reduce`.

        Reduction of a rotation ``p`` is a transformation of the form
        ``q = l * p * r``, where ``l`` and ``r`` are chosen from `left` and
        `right` respectively, such that rotation ``q`` has the smallest
        magnitude.

        If `left` and `right` are rotation groups representing symmetries of
        two objects rotated by ``p``, then ``q`` is the rotation of the
        smallest magnitude to align these objects considering their symmetries.

        Parameters
        ----------
        left : Rotation, optional
            Object containing the left rotation(s). Default value (None)
            corresponds to the identity rotation.
        right : Rotation, optional
            Object containing the right rotation(s). Default value (None)
            corresponds to the identity rotation.
        return_indices : bool, optional
            Whether to return the indices of the rotations from `left` and
            `right` used for reduction.

        Returns
        -------
        reduced : Rotation
            Object containing reduced rotations.
        left_best, right_best: integer ndarray
            Indices of elements from `left` and `right` used for reduction.
        """
        if return_indices:
            rot, left_idx, right_idx = \
                self._rot.reduce(left, right, return_indices)
            return self._from_scipy_rotation(rot), left_idx, right_idx

        rot = self._rot.reduce(left, right, return_indices)
        return self._from_scipy_rotation(rot)


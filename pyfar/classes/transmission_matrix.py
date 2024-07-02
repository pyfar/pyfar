"""
The following introduces the :py:class:`~TransmissionMatrix` class available in pyfar.

Transmission matrices (short T-matrices) are a convenient representation of
`Two-ports` (or `Quadrupoles`). These can represent systems from various fields,
e.g. electrical circuits, mechanical vibration, (acoustic) transmission lines.

Systems can be cascaded my multiplying consecutive T-matrices. Furthermore,
properties like input impedance of transfer functions can directly be derived.
"""
from __future__ import annotations
import numpy as np
from pyfar.classes.audio import FrequencyData


class TransmissionMatrix(FrequencyData):
    r"""Class representing a transmission matrix

    This implementation is based on a paper by Lampton [#]_ and uses the ABCD-
    representation. A single T-matrix is a (2x2)-matrix of the form:

    .. math::
        T = \begin{bmatrix}
                A & B \\
                C & D
            \end{bmatrix}

    This class represents frequency-dependent matrix of a multi-dimensional form,
    i.e. A, B, C and D can be matrices as well. For this purpose, it is derived
    from the :py:class:`~pyfar.classes.audio.FrequencyData` class. In the easiest
    case, each matrix entry (A,B,C,D) is a single-channel FrequencyData object (i.e.
    a vector depending on frequency). However, additional dimensions can be used to
    represent additional variables (e.g. multiple layouts of an electrical circuit).

    Notes
    -----
    This class is derived from the :py:class:`~pyfar.classes.audio.FrequencyData` but
    has a special constraint: The input data must match a shape like (..., 2, 2, N), so
    the resulting cshape is (...,2,2). The last axis refers to the frequency, and the
    two axes before to the ABCD-Matrix (which is a 2x2 matrix). For example,
    `obj[...,0,0]` returns the A-entry as
    :py:class:`~pyfar.classes.audio.FrequencyData`.

    Another point is the numerical handling when deriving input/output impedance or
    transfer functions (see :py:func:`~input_impedance`, :py:func:`~output_impedance`,
    :py:func:`~transfer_function_quantity1`, :py:func:`~transfer_function_quantity2`):
    In the respective equations, the load impedance Zl is usually multiplied. This can
    lead to numerical problems for Zl = inf. Thus, Zl is applied in the form of 1/Zl (as
    admittance) in this case to avoid such problems. Nevertheless, there are additional
    cases where certain entries of the T-matrix being zero might still lead to the main
    denominator becoming zero. For these cases, the denominator is set to eps.

    References
    ----------
    .. [#] M. Lampton. “Transmission Matrices in Electroacoustics". Acustica. Vol. 39,
           pp. 239-251. 1978.

    """

    def __init__(self, data, frequencies, comment = ""):
        """Create TransmissionMatrix with data, and frequencies.

        To ensure handing the correct data, it is recommended to use the
        :py:func:`~from_abcd` method to create objects.

        Parameters
        ----------
        data : array, double
            Raw data in the frequency domain. The memory layout of Data is 'C'.
            In contrast to the :py:class:`~pyfar.classes.audio.FrequencyData` class,
            data must have a shape of the form (..., 2, 2, N), e.g. (2, 2, 1024) or
            (3, 2, 2, 1024). In those examples 1024 refers to the number of frequency
            bins and the two dimensions before that to the (2x2) ABCD matrices.
            Supported data types are ``int``, ``float`` or ``complex``, where
            ``int`` is converted to ``float``.
        frequencies : array, double
            Frequencies of the data in Hz. The number of frequencies must match
            the size of the last dimension of data.
        comment : str, optional
            A comment related to the data. The default is ``""``, which
            initializes an empty string.

        """
        shape = np.shape(data)
        n_dim = len(shape)
        if n_dim < 3 or shape[-3] != 2 or shape[-2] != 2:
            raise ValueError("'data' must have a shape like [..., 2, 2, N]"
                             ", e.g. [2, 2, 100].")

        super().__init__(data, frequencies, comment)

    @classmethod
    def from_abcd(cls, A,B,C,D, frequencies = None):
        """Create a TransmissionMatrix object from A-, B-, C-, D-data, and frequencies.

        Parameters
        ----------
        A, B, C, D : FrequencyData, array, double
            Raw data for the matrix entries A, B, C and D. Must use the same data type
            for all inputs.
        frequencies : array, double, None
            Frequencies of the data in Hz. This is optional if using the FrequencyData
            type for A, B, C, D.

        """
        if (
            not isinstance(B, type(A))
            or not isinstance(C, type(A))
            or not isinstance(D, type(A))
        ):
            raise ValueError("A-,B-,C- and D-Matrices must be of the same type.")

        if isinstance(A, FrequencyData):
            frequencies = A.frequencies
            A = A.freq
            B = B.freq
            C = C.freq
            D = D.freq
        if frequencies is None:
            raise ValueError("'frequencies' must be specified if not using "
                             "'FrequencyData' objects as input.")

        data = np.array([[A, B], [C, D]])

        # Switch dimension order so that T matrices refer to
        # third and second last dimension (axes -3 and -2)
        order = np.array(range(data.ndim))
        order = np.roll(order, -2)  # Now T-axes indices are [-2, -1]; freq axis at [-3]
        order[[-1, -3]] = order[[-3, -1]]  # Correct freq axis index
        order[[-3, -2]] = order[[-2, -3]]  # Correct order of T-matrix dimensions
        data = np.permute_dims(data, order)

        return cls(data, frequencies)


    def cascade(self, t_matrix: TransmissionMatrix):
        """Cascades two systems (T-matrices) by doing a matrix multiplication

        Parameters
        ----------
        t_matrix : The other TransmissionMatrix to perform the matrix multiplication.

        Returns
        -------
        tmat_out : TransmissionMatrix
            The index for the given frequency. If the input was an array like,
            a numpy array of indices is returned.

        Notes
        -----
        An easier approach, especially to cascade multiple systems, is using the
        @-operator, e.g.::

            T = T1 @ T2 @ T3 ...

        See also Equation (2-3) in Reference [1]_.

        """
        return self @ t_matrix

    @property
    def abcd_axes(self):
        """The indices of the axes referring to the transmission matrix

        This relates to the axes with respect to the full data set
        (including the frequency-axis).

        """
        return (-3, -2)

    @property
    def abcd_caxes(self):
        """The indices of the channel axes referring to the transmission matrix

        This relates to the axes with respect to channel-related data set
        (excluding the frequency-axis).

        """
        return (-2, -1)

    @property
    def abcd_cshape(self):
        """The channel shape of the transmission matrix entries (A, B, C, D)

        This is the same as 'cshape' without the last two elements, but atleast (1,).

        """
        return self.A.cshape

    @property
    def A(self) -> FrequencyData:
        """Returns the (potentially multi-dimensional) A entry of the T-matrix."""
        return self[..., 0, 0, :]

    @property
    def B(self) -> FrequencyData:
        """Returns the (potentially multi-dimensional) B entry of the T-matrix."""
        return self[..., 0, 1, :]

    @property
    def C(self) -> FrequencyData:
        """Returns the (potentially multi-dimensional) C entry of the T-matrix."""
        return self[..., 1, 0, :]

    @property
    def D(self) -> FrequencyData:
        """Returns the (potentially multi-dimensional) D entry of the T-matrix."""
        return self[..., 1, 1, :]

    def _check_for_inf(self, Zl: complex | FrequencyData) -> bool:
        """Check given load impedance for np.inf values

        Returns
        -------
        idx_inf : logical array
            An np.ndarray of logicals pointing to elements referring to Zl = inf.
            The shape is broadcasted to self.A.freq so that it can be applied to
            the A-,B-,C-,D-entries.
        idx_inf_Zl : logical array
            An np.ndarray of logicals pointing to elements referring to Zl = inf.
            The shape refers to an indexable version of given Zl.
        Zl_indexable : An indexable version of Zl.

        """
        if isinstance(Zl, FrequencyData):
            Zl = Zl.freq
        elif np.isscalar(Zl):
            Zl = np.atleast_1d(Zl)

        target_shape = list(self.abcd_cshape) + [self.n_bins]
        idx_inf_Zl = Zl == np.inf
        if target_shape is not None:
            idx_inf = np.broadcast_to(idx_inf_Zl, target_shape)

        return idx_inf, idx_inf_Zl, Zl

    def input_impedance(self, Zl: complex | FrequencyData) -> FrequencyData:
        """Calculates the input impedance given the load impedance Zl at the output.

        Two-port representation::

                    o---xxxxxxxxx---o
                        x       x   |
            Zin-->      x       x   Zl
                        x       x   |
                    o---xxxxxxxxx---o

        See Equation (2-6) in Reference [1]_.

        Parameters
        ----------
        Zl : scalar | FrequencyData
            The load impedance data as scalar or FrequencyData. The shape must match
            the entries of the T-matrix, i.e. shape(tmat.A.freq) == shape(Zl.freq) or
            must be broadcastable.

        Returns
        -------
        Zout : FrequencyData
            A FrequencyData object with the resulting output impedance. The shape is
            identical to the entries of the T-matrix, i.e. shape(tmat.A) == shape(Zout).

        """
        nominator = (self.A * Zl + self.B)
        denominator = (self.C * Zl + self.D)

        # Admittance form for Zl = inf
        idx_inf, __, __ = self._check_for_inf(Zl)
        nominator.freq[idx_inf] = (self.A + self.B / Zl).freq[idx_inf]
        denominator.freq[idx_inf] = (self.C + self.D / Zl).freq[idx_inf]

        # Avoid cases where denominator is zero, examples
        # Zl = inf & C = 0; Zl = 0 & D = 0
        denominator.freq[denominator.freq == 0] = np.finfo(float).eps
        return nominator / denominator

    def output_impedance(self, Zl: complex | FrequencyData) -> FrequencyData:
        """Calculates the output impedance given the load impedance Zl at the input.

        Two-port representation::

                    o---xxxxxxxxx---o
                    |   x       x
                    Zl  x       x      <--Zout
                    |   x       x
                    o---xxxxxxxxx---o

        See Equation (2-6) in Reference [1]_.

        Parameters
        ----------
        Zl : scalar | FrequencyData
            The load impedance data as scalar or FrequencyData. The shape must match
            the entries of the T-matrix, i.e. shape(tmat.A.freq) == shape(Zl.freq) or
            must be broadcastable.

        Returns
        -------
        Zout : FrequencyData
            A FrequencyData object with the resulting output impedance. The shape is
            identical to the entries of the T-matrix, i.e. shape(tmat.A) == shape(Zout).

        """
        nominator = (self.D * Zl + self.B)
        denominator = (self.C * Zl + self.A)

        # Admittance form for Zl = inf
        idx_inf, __, __ = self._check_for_inf(Zl)
        nominator.freq[idx_inf] = (self.D + self.B / Zl).freq[idx_inf]
        denominator.freq[idx_inf] = (self.C + self.A / Zl).freq[idx_inf]

        # Avoid cases where denominator is zero, examples
        # Zl = 0 & A = 0; Zl = inf & C = 0
        denominator.freq[denominator.freq == 0] = np.finfo(float).eps
        return nominator / denominator

    def transfer_function_quantity1(self, Zl: complex | FrequencyData) -> FrequencyData:
        """Returns the transfer function of the first quantity (output/input).

        The TF is calculated based on the load impedance at the output.
        The first quantity usually refers to voltage, force, pressure, etc.
        See Equation (2-1) in Reference [1]_: Defined as e2/e1 using i2 = e2/Zl.

        Parameters
        ----------
        Zl : scalar | FrequencyData
            The load impedance data as scalar or FrequencyData. The shape must match
            the entries of the T-matrix, i.e. shape(tmat.A.freq) == shape(Zl.freq) or
            must be broadcastable.

        Returns
        -------
        TF : FrequencyData
            A FrequencyData object with the resulting transfer function. The shape is
            identical to the entries of the T-matrix, i.e. shape(tmat.A) == shape(TF).

        """
        idx_inf, __, __ = self._check_for_inf(Zl)
        denominator = (self.A * Zl + self.B)
        nominator = Zl*FrequencyData(np.ones_like(denominator.freq), self.frequencies)

        # Admittance form for Zl = inf
        nominator.freq[idx_inf] = 1
        denominator.freq[idx_inf] = (self.A + self.B / Zl).freq[idx_inf]

        # Avoid cases where denominator is zero, examples
        # Zl = 0 & B = 0; Zl = inf & A = 0
        denominator.freq[denominator.freq == 0] = np.finfo(float).eps
        return nominator / denominator

    def transfer_function_quantity2(self, Zl: complex | FrequencyData) -> FrequencyData:
        """Returns the transfer function of the second quantity (output/input).

        The TF is calculated based on the load impedance at the output. The second
        quantity usually refers to some flow or movement, e.g. current, velocity,...
        See Equation (2-1) in Reference [1]_: Defined as i2/i1 using e2 = i2*Zl.

        Parameters
        ----------
        Zl : scalar | FrequencyData
            The load impedance data as scalar or FrequencyData. The shape must match
            the entries of the T-matrix, i.e. shape(tmat.A.freq) == shape(Zl.freq) or
            must be broadcastable.

        Returns
        -------
        TF : FrequencyData
            A FrequencyData object with the resulting transfer function. The shape is
            identical to the entries of the T-matrix, i.e. shape(tmat.A) == shape(TF).

        """
        denominator = (self.C * Zl + self.D)
        nominator = FrequencyData(np.ones_like(denominator.freq), self.frequencies)

        # Admittance form for Zl = inf
        idx_inf_target, idx_inf_Zl, Zl_indexable = self._check_for_inf(Zl)
        nominator.freq[idx_inf_target] = 1 / Zl_indexable[idx_inf_Zl]
        denominator.freq[idx_inf_target] = (self.C + self.D / Zl).freq[idx_inf_target]

        # Avoid cases where denominator is zero, examples
        # Zl = 0 & D = 0; Zl = inf & C = 0
        denominator.freq[denominator.freq == 0] = np.finfo(float).eps
        return nominator / denominator

    # def transfer_function_quantity1_to_quantity2(self, Zl):
    #     pass
    # def transfer_function_quantity2_to_quantity1(self, Zl):
    #     pass

    @classmethod
    def _create_abcd_matrix_broadcast(cls, abcd_data: FrequencyData,
                                      abcd_cshape : tuple):
        if abcd_data.cdim != 2 or abcd_data.cshape != (2,2):
            raise ValueError("'abcd_data' must have a cshape of (2,2).")

        if abcd_cshape != ():
            abcd_data.freq = np.broadcast_to(abcd_data.freq,
                                np.append(abcd_cshape, abcd_data.freq.shape))
        return cls(abcd_data.freq, abcd_data.frequencies)

    @classmethod
    def create_identity(cls, frequencies, abcd_cshape = ()):
        """Creates an object with identity matrix entries (bypass).

        See Equation (2-7) in Table I of Reference [1]_.

        Parameters
        ----------
        frequencies : array_like
            The frequency sampling points in Hz.
        abcd_cshape : tuple | int, optional
            Shape of additional dimensions besides (2,2,num_bins).
            Per default, no additional dimensions are created.

        Returns
        -------
        cls : TransmissionMatrix
            A TransmissionMatrix object that contains 2x2 identity matrices.

        """
        identity = np.array([[1,0],[0,1]])
        identity_3d = np.repeat(identity[:, :, np.newaxis], len(frequencies), axis = 2)
        return cls._create_abcd_matrix_broadcast(
            FrequencyData(identity_3d, frequencies), abcd_cshape)

    @classmethod
    def create_series_impedance(cls, impedance : FrequencyData, abcd_cshape = ()):
        """Creates a transmission matrix representing a series impedance.

        This means the impedance is connected in series with a potential load impedance.
        See Equation (2-8) in Table I of Reference [1]_.

        Parameters
        ----------
        impedance : FrequencyData
            The impedance data of the series impedance.
        abcd_cshape : tuple | int, optional
            Shape of additional dimensions besides (2,2,num_bins).
            Per default, no additional dimensions are created.

        Returns
        -------
        cls : TransmissionMatrix
            A TransmissionMatrix representing a series impedance.

        """

        if impedance.cshape != (1,):
            raise ValueError("Number of channels for 'impedance' must be 1.")
        Z = impedance.freq[0]
        ones, zeros = np.ones_like(Z), np.zeros_like(Z)
        series_Z_abcd = FrequencyData([[ones, Z],[zeros, ones]], impedance.frequencies)
        return cls._create_abcd_matrix_broadcast(series_Z_abcd, abcd_cshape)

    @classmethod
    def create_shunt_admittance(cls, admittance : FrequencyData, abcd_cshape = ()):
        """Creates a transmission matrix representing a shunt admittance (parallel connection).

        In this case, the impedance (= 1 / admittance) is connected in parallel
        with a potential load impedance.
        See Equation (2-9) in Table I of Reference [1]_.

        Parameters
        ----------
        admittance : FrequencyData
            The admittance data of the element connected in parallel.
        abcd_cshape : tuple | int, optional
            Shape of additional dimensions besides (2,2,num_bins).
            Per default, no additional dimensions are created.

        Returns
        -------
        cls : TransmissionMatrix
            A TransmissionMatrix representing a parallel connection.

        """ # noqa: E501
        if admittance.cshape != (1,):
            raise ValueError("Number of channels for 'admittance' must be 1.")
        Y = admittance.freq[0]
        ones, zeros = np.ones_like(Y), np.zeros_like(Y)
        series_Z_abcd = FrequencyData([[ones, zeros],[Y, ones]], admittance.frequencies)
        return cls._create_abcd_matrix_broadcast(series_Z_abcd, abcd_cshape)

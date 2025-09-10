"""
The following introduces the :py:class:`~TransmissionMatrix` class available
in pyfar.

Transmission matrices (short T-matrices) are a convenient representation of
`Two-ports` or `Quadrupoles`. These can represent systems from various
fields, e.g. electrical circuits, mechanical vibration, (acoustic) transmission
lines.

System properties like input impedance or transfer functions can directly be
derived from a T-matrix. Furthermore, systems can be cascaded by multiplying
consecutive T-matrices simply using the ``@`` operator:

>>> import numpy as np
>>> import pyfar as pf
>>> frequencies = (100,200,300)
>>>
>>> # T-matrix with arbitrary data
>>> A = np.ones(len(frequencies))
>>> (B,C,D) = (A+1,A+2,A+3)
>>> tmat = pf.TransmissionMatrix.from_abcd(A,B,C,D,frequencies)
>>>
>>> # T-matrix of a bypass system
>>> tmat_bypass = pf.TransmissionMatrix.create_identity(frequencies)
>>>
>>> # Cascade
>>> tmat_out = tmat @ tmat_bypass
>>> tmat_out.freq == tmat.freq

"""

from __future__ import annotations
from numbers import Number  # required for Python <= 3.9
import numpy as np
import numpy.testing as npt
from pyfar.classes.audio import FrequencyData
from pyfar.constants import reference_air_impedance, reference_speed_of_sound
from numbers import Number


class TransmissionMatrix(FrequencyData):
    r"""Class representing a transmission matrix.

    This implementation is based on a paper by Lampton [#]_ and uses the ABCD-
    representation. A single T-matrix is a (2x2)-matrix of the form:

    .. math::
        T = \begin{bmatrix}
                A & B \\
                C & D
            \end{bmatrix}

    This class is derived from :py:class:`~pyfar.classes.audio.FrequencyData`,
    representing a frequency-dependent matrix of a multi-dimensional form
    In the easiest case, each matrix entry (A,B,C,D) is a single-channel
    FrequencyData object (i.e. a vector depending on frequency). However,
    additional dimensions can be used to represent additional variables (e.g.
    multiple layouts of an electrical circuit).

    Notes
    -----
    This class is derived from
    :py:class:`~pyfar.classes.audio.FrequencyData` but has a special
    constraint: The underlying frequency domain data must match a shape like
    (..., 2, 2, N). The last axis refers to the frequency, and the two axes
    before to the ABCD-Matrix (which is a 2x2 matrix), so the resulting cshape
    is (...,2,2).

    Another point is the numerical handling when deriving input/output
    impedance or transfer functions (see :py:func:`~input_impedance`,
    :py:func:`~output_impedance`, :py:func:`~transfer_function`):
    In the respective equations, the load impedance Zl is usually multiplied.
    However, Zl is applied in the form of 1/Zl (as admittance) for Zl = inf
    to avoid numerical problems.
    There are additional cases where certain entries of the
    T-matrix being zero might still lead to the main denominator becoming zero.
    For these cases, the denominator is set to eps.

    References
    ----------
    .. [#] M. Lampton. “Transmission Matrices in Electroacoustics". Acustica.
           Vol. 39, pp. 239-251. 1978.

    """

    def __init__(self, data, frequencies, comment=""):
        """Initialize TransmissionMatrix with data, and frequencies.

        This should not be used directly. Instead use :py:func:`~from_abcd`.
        or :py:func:`~from_tmatrix`.

        """
        shape = np.shape(data)
        n_dim = len(shape)
        if n_dim < 3 or shape[-3] != 2 or shape[-2] != 2:
            raise ValueError(
                "'data' must have a shape like "
                "(..., 2, 2, n_bins), e.g. (2, 2, 100)."
            )

        super().__init__(data, frequencies, comment)

    @classmethod
    def from_tmatrix(cls, data, frequencies, comment=""):
        """Create TransmissionMatrix using data in T-matrix shape
        and frequencies.

        For using individual objects for A, B, C, D matrix entries,
        see :py:func:`~from_abcd`.

        Parameters
        ----------
        data : array_like, double
            Raw data in the frequency domain.
            In contrast to the :py:class:`~pyfar.classes.audio.FrequencyData`
            class, data must have a shape of the form (..., 2, 2, N), e.g.
            (2, 2, 1024) or (3, 2, 2, 1024). In those examples 1024 refers to
            the number of frequency bins and the two dimensions before that to
            the (2x2) ABCD matrices. Supported data types are ``int``,
            ``float`` or ``complex``, where ``int`` is converted to ``float``.
        frequencies : array_like, double
            Frequencies of the data in Hz. The number of frequencies must match
            the size of the last dimension of data.
        comment : str, optional
            A comment related to the data. The default is ``""``, which
            initializes an empty string.

        Examples
        --------
        >>> import pyfar as pf
        >>> frequencies = (100,200,300)
        >>> data = np.ones( (2, 2, len(frequencies)) )
        >>> tmat = pf.TransmissionMatrix.from_tmatrix(data, frequencies)

        >>> data = np.ones( (3, 2, 2, len(frequencies)) )
        >>> tmat = pf.TransmissionMatrix.from_tmatrix(data, frequencies)

        """
        return cls(data, frequencies, comment)

    def _tmat_from_abcd(A, B, C, D):
        """Create a T-matrix from A-, B-, C-, D-data."""
        if not (
            (isinstance(A, Number) or isinstance(A, complex))
            and (isinstance(B, Number) or isinstance(B, complex))
            and (isinstance(C, Number) or isinstance(C, complex))
            and (isinstance(D, Number) or isinstance(D, complex))
        ):
            raise TypeError(
                "A, B, C, and D must be scalars (real or complex)."
            )
        return np.array([[A, B], [C, D]])

    @classmethod
    def from_abcd(cls, A, B, C, D, frequencies=None):
        """Create a TransmissionMatrix object from A-, B-, C-, D-data, and
        frequencies.

        Parameters
        ----------
        A : FrequencyData, array_like, double
            Raw data for the matrix entries A. The data need to match the
            B, C and D entry or be broadcastable into one ``shape``.
        B : FrequencyData, array_like, double
            See A.
        C : FrequencyData, array_like, double
            See A.
        D : FrequencyData, array_like, double
            See A.
        frequencies : array, double, None
            Frequencies of the data in Hz. This is optional if using the
            FrequencyData type for A, B, C, D.

        Examples
        --------
        >>> import numpy as np
        >>> import pyfar as pf
        >>> frequencies = (100,200,300)

        >>> # From np.array
        >>> A = np.ones(len(frequencies))
        >>> (B,C,D) = (A+1, A+2, A+3)
        >>> tmat = pf.TransmissionMatrix.from_abcd(A,B,C,D, frequencies)
        >>> tmat

        >>> # From FrequencyData objects
        >>> A = pf.FrequencyData( A, frequencies )
        >>> (B,C,D) = (A+1, A+2, A+3)
        >>> tmat = pf.TransmissionMatrix.from_abcd(A,B,C,D)
        >>> tmat

        >>> # Data with higher abcd dimension
        >>> A = np.ones( (3, 4, len(frequencies)) )
        >>> (B,C,D) = (A+1, A+2, A+3)
        >>> tmat = pf.TransmissionMatrix.from_abcd(A,B,C,D, frequencies)
        >>> tmat

        """
        num_freqdata = 0
        for obj in (A, B, C, D):
            if isinstance(obj, FrequencyData):
                num_freqdata = num_freqdata + 1

        if num_freqdata == 4:
            frequencies = A.frequencies
            for obj in (B, C, D):  # Frequency bins must match
                npt.assert_allclose(frequencies, obj.frequencies, atol=1e-15)
            (A, B, C, D) = (A.freq, B.freq, C.freq, D.freq)
        elif num_freqdata != 0:
            raise ValueError(
                "If using FrequencyData objects, all matrix entries "
                "A, B, C, D, must be FrequencyData objects."
            )
        (A, B, C, D) = (
            np.atleast_1d(np.asanyarray(A, dtype=np.float64)),
            np.atleast_1d(np.asanyarray(B, dtype=np.float64)),
            np.atleast_1d(np.asanyarray(C, dtype=np.float64)),
            np.atleast_1d(np.asanyarray(D, dtype=np.float64)),
        )
        if frequencies is None:
            raise ValueError(
                "'frequencies' must be specified if not using "
                "'FrequencyData' objects as input."
            )
        # broadcast shapes
        shape = np.broadcast_shapes(
            np.array(A).shape,
            np.array(B).shape,
            np.array(C).shape,
            np.array(D).shape,
        )
        A = np.broadcast_to(A, shape)
        B = np.broadcast_to(B, shape)
        C = np.broadcast_to(C, shape)
        D = np.broadcast_to(D, shape)

        data = np.array([[A, B], [C, D]])
        # Switch dimension order so that T matrices refer to
        # third and second last dimension (axes -3 and -2)
        order = np.array(range(data.ndim - 1))
        order = np.roll(order, -2)
        order = np.append(order, data.ndim - 1)
        data = np.transpose(data, order)

        return cls(data, frequencies)

    @property
    def abcd_caxes(self):
        """The indices of the channel axes referring to the transmission
        matrix, namely (-2, -1).
        """
        return (-2, -1)

    @property
    def abcd_cshape(self):
        """The channel shape of the transmission matrix entries (A, B, C, D).

        This is the same as 'cshape' without the last two elements.
        As an exception, a matrix with cshape (2,2) will return
        (1,) as abcd_cshape.

        """
        abcd_cshape = self.cshape[:-2]
        if abcd_cshape == ():
            abcd_cshape = (1,)
        return abcd_cshape

    @property
    def A(self) -> FrequencyData:
        """A entry of the transmission matrix."""
        return FrequencyData(
            self.freq[..., 0, 0, :], self.frequencies, self.comment
        )

    @property
    def B(self) -> FrequencyData:
        """B entry of the transmission matrix."""
        return FrequencyData(
            self.freq[..., 0, 1, :], self.frequencies, self.comment
        )

    @property
    def C(self) -> FrequencyData:
        """C entry of the transmission matrix."""
        return FrequencyData(
            self.freq[..., 1, 0, :], self.frequencies, self.comment
        )

    @property
    def D(self) -> FrequencyData:
        """D entry of the transmission matrix."""
        return FrequencyData(
            self.freq[..., 1, 1, :], self.frequencies, self.comment
        )

    def _check_for_inf(self, Zl: complex | FrequencyData):
        """Check given load impedance for np.inf values.

        Returns
        -------
        idx_inf : logical array
            An np.ndarray of logicals pointing to elements referring to
            Zl = inf. The shape is broadcasted to self.A.freq so that it can be
            applied to the A-,B-,C-,D-entries.
        idx_inf_Zl : logical array
            An np.ndarray of logicals pointing to elements referring to
            Zl = inf. The shape refers to an indexable version of given Zl.
        Zl_indexable : np.ndarray
            An indexable version of Zl.

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
        r"""Calculates the input impedance given the load impedance Zl at the
        output.

        Two-port representation::

                    o---xxxxxxxxx---o
                        x       x   |
            Zin-->      x       x   Zl
                        x       x   |
                    o---xxxxxxxxx---o

        See Equation (2-6) in Reference [1]_:
        :math:`Z_\mathrm{in} = \frac{AZ_L + B}{CZ_L + D}`

        Parameters
        ----------
        Zl : scalar | FrequencyData
            The load impedance data as scalar or FrequencyData. In latter case,
            the shape must match the entries of the T-matrix, i.e.
            shape(tmat.A.freq) == shape(Zl.freq), or must be broadcastable.

        Returns
        -------
        Zout : FrequencyData
            A FrequencyData object with the resulting output impedance. The
            cshape is identical to the entries of the T-matrix, i.e.
            tmat.A.cshape == Zout.cshape.

        Example
        -------

        >>> import numpy as np
        >>> import pyfar as pf
        >>>
        >>> # Frequency-dependent load impedance
        >>> frequencies = (100,200,300)
        >>> load_impedance = pf.FrequencyData((0, 1, np.inf), frequencies)
        >>>
        >>> # T-Matrix for frequency-independent series impedance R = 1 Ohm
        >>> R = pf.FrequencyData((1,1,1), frequencies)
        >>> tmat = pf.TransmissionMatrix.create_series_impedance(R)
        >>>
        >>> # Expected result: (1+0, 1+1, 1+inf) = (1, 2, inf)
        >>> # Note, that due to numerical limitations infinite load will
        >>> # result in Zin > 1e15.
        >>> Zin = tmat.input_impedance(load_impedance)
        >>> Zin.freq

        """
        nominator = self.A * Zl + self.B
        denominator = self.C * Zl + self.D

        # Admittance form for Zl = inf
        idx_inf, __, __ = self._check_for_inf(Zl)
        nominator.freq[idx_inf] = (self.A + self.B / Zl).freq[idx_inf]
        denominator.freq[idx_inf] = (self.C + self.D / Zl).freq[idx_inf]

        # Avoid cases where denominator is zero, examples
        # Zl = inf & C = 0; Zl = 0 & D = 0
        denominator.freq[denominator.freq == 0] = np.finfo(float).eps
        return nominator / denominator

    def output_impedance(self, Zl: complex | FrequencyData) -> FrequencyData:
        r"""Calculates the output impedance given the load impedance Zl at the
        input.

        Two-port representation::

                    o---xxxxxxxxx---o
                    |   x       x
                    Zl  x       x      <--Zout
                    |   x       x
                    o---xxxxxxxxx---o

        See Equation (2-6) in Reference [1]_:
        :math:`Z_\mathrm{out} = \frac{DZ_L + B}{CZ_L + A}`

        For a code example, see :py:func:`~input_impedance` and exchange
        respective method call with `output_impedance`.

        Parameters
        ----------
        Zl : scalar | FrequencyData
            The load impedance data as scalar or FrequencyData. In latter case,
            the shape must match the entries of the T-matrix, i.e.
            shape(tmat.A.freq) == shape(Zl.freq), or must be broadcastable.

        Returns
        -------
        Zout : FrequencyData
            A FrequencyData object with the resulting output impedance. The
            cshape is identical to the entries of the T-matrix, i.e.
            tmat.A.cshape == Zout.cshape.

        """
        nominator = self.D * Zl + self.B
        denominator = self.C * Zl + self.A

        # Admittance form for Zl = inf
        idx_inf, __, __ = self._check_for_inf(Zl)
        nominator.freq[idx_inf] = (self.D + self.B / Zl).freq[idx_inf]
        denominator.freq[idx_inf] = (self.C + self.A / Zl).freq[idx_inf]

        # Avoid cases where denominator is zero, examples
        # Zl = 0 & A = 0; Zl = inf & C = 0
        denominator.freq[denominator.freq == 0] = np.finfo(float).eps
        return nominator / denominator

    def transfer_function(
        self, quantity_indices, Zl: complex | FrequencyData
    ) -> FrequencyData:
        r"""Returns the transfer function (output/input) for specified
        quantities and a given load impedance.

        The transfer function is the relation between an output and input
        quantity of modelled two-port and depends on the load impedance at the
        output. Since there are two quantities at input and output
        respectively, four transfer functions exist in total. The first usually
        refers to the "voltage-like" quantity (:math:`Q_1`) whereas the second
        refers to the "current-like" quantity (:math:`Q_2`).

        The transfer functions can be derived from Equation (2-1)
        in Reference [1]_:

        .. math::
            Q_{1,\mathrm{in}} = AQ_{1,\mathrm{out}} + BQ_{2,\mathrm{in}}

            Q_{2,\mathrm{in}} = CQ_{1,\mathrm{out}} + DQ_{2,\mathrm{in}}

        The four transfer functions are defined as:

        * :math:`Q_{1,\mathrm{out}} / Q_{1,\mathrm{in}}` using
          :math:`Q_{2,\mathrm{out}} = Q_{1,\mathrm{out}}/Z_L`
        * :math:`Q_{2,\mathrm{out}} / Q_{1,\mathrm{in}} =
          Q_{1,\mathrm{out}} / Q_{1,\mathrm{in}} \cdot \frac{1}{Z_L}`
        * :math:`Q_{2,\mathrm{out}} / Q_{2,\mathrm{in}}` using
          :math:`Q_{1,\mathrm{out}} = Q_{2,\mathrm{out}}\cdot Z_L`
        * :math:`Q_{1,\mathrm{out}} / Q_{2,\mathrm{in}} =
          Q_{2,\mathrm{out}} / Q_{2,\mathrm{in}} \cdot Z_L`

        Parameters
        ----------
        quantity_indices : array_like (int, int)
            Array-like object with two integer elements referring to the
            indices of the utilized quantity at the output (first integer)
            and input (second integer). For example, (1,0) refers to the
            transfer function :math:`Q_{2,\mathrm{out}} / Q_{1,\mathrm{in}}`.
        Zl : scalar | FrequencyData
            The load impedance data as scalar or FrequencyData. In latter case,
            the shape must match the entries of the T-matrix, i.e.
            shape(tmat.A.freq) == shape(Zl.freq), or must be broadcastable.

        Returns
        -------
        transfer_function : FrequencyData
            A FrequencyData object with the resulting transfer function. The
            cshape is identical to the entries of the T-matrix, i.e.
            tmat.A.cshape == transfer_function.cshape.

        """
        is_scalar = np.isscalar(quantity_indices)
        quantity_indices = np.array(quantity_indices)
        is_numeric = np.issubdtype(quantity_indices.dtype, np.number)
        if is_scalar or not is_numeric or not len(quantity_indices) == 2:
            raise ValueError(
                "'quantity_indices' must be an array-like type "
                "with two numeric elements."
            )
        if not all(
            np.logical_or(quantity_indices == 0, quantity_indices == 1)
        ):
            raise ValueError(
                "'quantity_indices' must contain two integers "
                "between 0 and 1."
            )

        if quantity_indices[0] == 0 and quantity_indices[1] == 0:
            return self._transfer_function_q1q1(Zl)
        if quantity_indices[0] == 1 and quantity_indices[1] == 1:
            return self._transfer_function_q2q2(Zl)
        if quantity_indices[0] == 1 and quantity_indices[1] == 0:
            return self._transfer_function_q2q1(Zl)
        if quantity_indices[0] == 0 and quantity_indices[1] == 1:
            return self._transfer_function_q1q2(Zl)

    def _transfer_function_q1q1(
        self, Zl: complex | FrequencyData
    ) -> FrequencyData:
        """Returns the first quantity's transfer function (Q1_out/Q1_in)."""
        idx_inf, __, __ = self._check_for_inf(Zl)
        denominator = self.A * Zl + self.B
        nominator = Zl * FrequencyData(
            np.ones_like(denominator.freq), self.frequencies
        )

        # Admittance form for Zl = inf
        nominator.freq[idx_inf] = 1
        denominator.freq[idx_inf] = (self.A + self.B / Zl).freq[idx_inf]

        # Avoid cases where denominator is zero, examples
        # Zl = 0 & B = 0; Zl = inf & A = 0
        denominator.freq[denominator.freq == 0] = np.finfo(float).eps
        return nominator / denominator

    def _transfer_function_q2q1(
        self, Zl: complex | FrequencyData
    ) -> FrequencyData:
        """Returns the transfer function Q2_out / Q1_in."""
        denominator = self.A * Zl + self.B
        nominator = FrequencyData(
            np.ones_like(denominator.freq), self.frequencies
        )

        # In cases where the denominator is zero, e.g. Zl = 0 & B = 0,
        # are related to short-circuated outputs (undefined current) => NaN
        denominator.freq[denominator.freq == 0] = np.nan
        return nominator / denominator

    def _transfer_function_q2q2(
        self, Zl: complex | FrequencyData
    ) -> FrequencyData:
        """Returns the second quantity's transfer function (Q2_out/Q2_in)."""
        denominator = self.C * Zl + self.D
        nominator = FrequencyData(
            np.ones_like(denominator.freq), self.frequencies
        )

        # Admittance form for Zl = inf
        idx_inf, idx_inf_Zl, Zl_indexable = self._check_for_inf(Zl)
        nominator.freq[idx_inf] = 1 / Zl_indexable[idx_inf_Zl]
        denominator.freq[idx_inf] = (self.C + self.D / Zl).freq[idx_inf]

        # Avoid cases where denominator is zero, examples
        # Zl = 0 & D = 0; Zl = inf & C = 0
        denominator.freq[denominator.freq == 0] = np.finfo(float).eps
        return nominator / denominator

    def _transfer_function_q1q2(
        self, Zl: complex | FrequencyData
    ) -> FrequencyData:
        """Returns the transfer function Q1_out / Q2_in."""
        denominator = self.C * Zl + self.D
        nominator = Zl * FrequencyData(
            np.ones_like(denominator.freq), self.frequencies
        )

        # Admittance form for Zl = inf
        idx_inf, __, __ = self._check_for_inf(Zl)
        nominator.freq[idx_inf] = 1
        denominator.freq[idx_inf] = (self.C + self.D / Zl).freq[idx_inf]

        # In cases where the denominator is zero, e.g. Zl = inf & C = 0,
        # are related to non-physical cases (e.g. undefined current/voltage)
        # => NaN
        denominator.freq[denominator.freq == 0] = np.nan
        return nominator / denominator

    @staticmethod
    def create_identity(frequencies=None):
        r"""Creates an object with identity matrix entries (bypass).

        See Equation (2-7) in Table I of Reference [1]_:

        .. math::
            T = \begin{bmatrix}
                1 & 0 \\
                0 & 1
                \end{bmatrix}

        Parameters
        ----------
        frequencies : None | array_like, optional
            The frequency sampling points in Hz. The default is `None` which
            will result in this function to return an np.ndarray instead of an
            TransmissionMatrix object.

        Returns
        -------
        tmat : np.ndarray | TransmissionMatrix
            If frequencies are specified, a TransmissionMatrix object that
            contains one 2x2 identity matrix per bin is returned. Otherwise,
            this returns a 2x2 np.ndarray.

        """
        if frequencies is None or len(frequencies) == 0:
            return np.eye(2)

        return TransmissionMatrix.from_abcd(
            np.ones_like(frequencies), 0, 0, 1, frequencies
        )

    @staticmethod
    def create_series_impedance(
        impedance: complex | FrequencyData,
    ) -> np.ndarray | TransmissionMatrix:
        r"""Creates a transmission matrix representing a series impedance.

        This means the impedance is connected in series with a potential load
        impedance. See Equation (2-8) in Table I of Reference [1]_:

        .. math::
            T = \begin{bmatrix}
                1 & Z \\
                0 & 1
                \end{bmatrix}

        Parameters
        ----------
        impedance : scalar | FrequencyData
            The impedance data of the series impedance.

        Returns
        -------
        tmat : np.ndarray | TransmissionMatrix
            A transmission matrix representing the series connection
            and can be cascaded with TransmissionMatrix objects.
            If a scalar was used as input a frequency-independent
            matrix is returned, namely an np.ndarray of shape (2,2).

        """
        if np.isscalar(impedance) and not isinstance(impedance, str):
            return TransmissionMatrix._tmat_from_abcd(1, impedance, 0, 1)

        if not isinstance(impedance, FrequencyData):
            raise ValueError(
                "'impedance' must be a "
                "numerical scalar or FrequencyData object."
            )
        return TransmissionMatrix.from_abcd(
            1, impedance.freq, 0, 1, impedance.frequencies
        )

    @staticmethod
    def create_shunt_admittance(
        admittance: complex | FrequencyData,
    ) -> np.ndarray | TransmissionMatrix:
        r"""Creates a transmission matrix representing a shunt admittance
        (parallel connection).

        In this case, the impedance (= 1 / admittance) is connected in parallel
        with a potential load impedance.
        See Equation (2-9) in Table I of Reference [1]_:

        .. math::
            T = \begin{bmatrix}
                1 & 0 \\
                Y & 1
                \end{bmatrix}

        Parameters
        ----------
        admittance : scalar | FrequencyData
            The admittance data of the element connected in parallel.

        Returns
        -------
        tmat : np.ndarray | TransmissionMatrix
            A transmission matrix representing a parallel connection
            and can be cascaded with TransmissionMatrix objects.
            If a scalar was used as input a frequency-independent
            matrix is returned, namely an np.ndarray of shape (2,2).

        """
        if np.isscalar(admittance) and not isinstance(admittance, str):
            return TransmissionMatrix._tmat_from_abcd(1, 0, admittance, 1)

        if not isinstance(admittance, FrequencyData):
            raise ValueError(
                "'admittance' must be a "
                "numerical scalar or FrequencyData object."
            )
        return TransmissionMatrix.from_abcd(
            1, 0, admittance.freq, 1, admittance.frequencies
        )

    @staticmethod
    def create_transformer(
        transducer_constant: float | int | FrequencyData,
    ) -> np.ndarray | TransmissionMatrix:
        r"""Creates a transmission matrix representing a transformer.

        See Equation (2-12) in Table I of Reference [1]_:

        .. math::
            T = \begin{bmatrix}
                N & 0 \\
                0 & 1/N
                \end{bmatrix}

        Parameters
        ----------
        transducer_constant : scalar | FrequencyData
            The transmission ratio with respect to voltage-like quantity,
            i.e. :math:`N=U_\mathrm{out}/U_\mathrm{in}`. If a scalar is given,
            i.e. a frequency-independent transformer matrix is requested, the
            return value will be a 2x2 np.ndarray instead.

        Returns
        -------
        tmat : np.ndarray | TransmissionMatrix
            A transmission matrix representing the transformer
            and can be cascaded with TransmissionMatrix objects.
            If a scalar was used as input a frequency-independent
            matrix is returned, namely an np.ndarray of shape (2,2).

        """
        if np.isscalar(transducer_constant) and not isinstance(
            transducer_constant, str
        ):
            return TransmissionMatrix._tmat_from_abcd(
                transducer_constant, 0, 0, 1 / transducer_constant
            )

        if not isinstance(transducer_constant, FrequencyData):
            raise ValueError(
                "'transducer_constant' must be a "
                "numerical scalar or FrequencyData object."
            )
        A = transducer_constant.freq
        D = (1 / transducer_constant).freq
        frequencies = transducer_constant.frequencies
        return TransmissionMatrix.from_abcd(A, 0, 0, D, frequencies)

    @staticmethod
    def create_gyrator(
        transducer_constant: complex | FrequencyData,
    ) -> np.ndarray | TransmissionMatrix:
        r"""Creates a transmission matrix representing a gyrator.

        The T-matrix is defined by a transducer constant (:math:`M`),
        see Equation (2-14) in Table I of Reference [1]_:

        .. math::
            T = \begin{bmatrix}
                0 & M \\
                1/M & 0
                \end{bmatrix}

        :math:`M` connects the first input and second output quantity (e.g.,
        :math:`U_\mathrm{out} = I_\mathrm{in} \cdot M`). A respective system
        with load :math:`Z_L` has the input impedance
        :math:`Z_\mathrm{in} = M^2 / Z_L`.

        Parameters
        ----------
        transducer_constant : scalar | FrequencyData
            The transducer constant :math:`M`. If a scalar is given,
            i.e. a frequency-independent transformer matrix is requested, the
            return value will be a 2x2 np.ndarray instead.

        Returns
        -------
        tmat : np.ndarray | TransmissionMatrix
            A the transmission matrix representing the gyrator and can be
            cascaded with TransmissionMatrix objects. If a scalar was used as
            input a frequency-independent matrix is returned, namely an
            np.ndarray of shape (2,2).

        """
        if np.isscalar(transducer_constant) and not isinstance(
            transducer_constant, str
        ):
            return TransmissionMatrix._tmat_from_abcd(
                0, transducer_constant, 1 / transducer_constant, 0
            )

        if not isinstance(transducer_constant, FrequencyData):
            raise ValueError(
                "'transducer_constant' must be a "
                "numerical scalar or FrequencyData object."
            )
        B = transducer_constant.freq
        C = (1 / transducer_constant).freq
        frequencies = transducer_constant.frequencies
        return TransmissionMatrix.from_abcd(0, B, C, 0, frequencies)

    @staticmethod
    def create_transmission_line(
            kl: FrequencyData,
            characteristic_impedance: Number | FrequencyData,
            ) -> TransmissionMatrix:
        r"""Create a transmission matrix representing a transmission line.

        The transmission matrix is calculated from the product :math:`kl` of
        the wavenumber :math:`k` and length :math:`l`,
        and characteristic impedance :math:`Z_0` following
        Equation (5-10) of Reference [1]_:

        .. math::
            T = \begin{bmatrix}
                \cos{kl} & j Z_0 \sin{kl} \\
                j/Z_0 \sin{kl} & \cos{kl}
                \end{bmatrix}

        The inputs need to be broadcastable into one `cshape`.

        Parameters
        ----------
        kl : FrequencyData
            The product of wavenumber and length,
            used as argument in the cosine and sine functions to calculate the
            entries of the matrix.
            If all imaginary parts of ``kl.freq`` are zero, the transmission
            matrix represents a lossless transmission line.
        characteristic_impedance : Number | FrequencyData
            The characteristic impedance of the transmission line.

        Returns
        -------
        tmat : TransmissionMatrix
            A transmission matrix representing a transmission line.


        Example
        -------
        Lossless acoustic duct with approximate ear canal dimensions.
        Plot its input impedance with rigid termination.

        .. plot::

            >>> import pyfar as pf
            >>> import numpy as np
            >>> # Transmission line parameters
            >>> frequencies = np.linspace(20, 20e3, 1000)
            >>> omega = 2*np.pi*frequencies
            >>> wavenumber = omega/pf.constants.reference_speed_of_sound
            >>> length = 30e-3
            >>> kl = pf.FrequencyData(wavenumber*length, frequencies)
            >>> cross_section = np.pi*(3.5e-3)**2
            >>> Z0 = pf.constants.reference_air_impedance/cross_section
            >>> # Create transmission matrix
            >>> T = pf.TransmissionMatrix.create_transmission_line(kl, Z0)
            >>> # Plot input impedance with rigid termination
            >>> ax = pf.plot.freq(T.input_impedance(np.inf))
            >>> ax.set_ylim(100, 200)

        """
        if not isinstance(kl, FrequencyData):
            raise ValueError("The input kl must be a FrequencyData object.")
        if isinstance(characteristic_impedance, FrequencyData):
            if not np.allclose(
                    kl.frequencies,
                    characteristic_impedance.frequencies, atol=1e-15):
                raise ValueError(
                    "The frequencies do not match.")
            else:
                characteristic_impedance = characteristic_impedance.freq
        elif not isinstance(characteristic_impedance, Number):
            raise ValueError(
                "characteristic impedance must be a Number or " \
                "FrequencyData object.")
        # broadcastable shapes are checked by from_abcd

        A = np.cos(kl.freq)
        B = 1j*characteristic_impedance*np.sin(kl.freq)
        C = 1j/characteristic_impedance*np.sin(kl.freq)
        D = A.copy()

        return TransmissionMatrix.from_abcd(
            A, B, C, D, kl.frequencies)

    @staticmethod
    def _calculate_horn_geometry_parameters(
        S0: Number,
        S1: Number,
        L: Number,
    ) -> tuple[Number, Number, Number]:
        """Calculate the geometry parameters of a conical horn.

        Parameters
        ----------
        S0 : float
            Cross-sectional area at the narrow end of the horn.
        S1 : float
            Cross-sectional area at the wide end of the horn.
        L : float
            Length of the horn.

        Returns
        -------
        tuple[float, float, float]
            A tuple containing the area constant Omega, the distance a from the
            narrow end to the virtual apex of the cone, and the distance b from
            the wide end to the same virtual apex.
        """
        if not isinstance(S0, Number) or isinstance(S0, complex) or S0 <= 0:
            raise ValueError("The input S0 must be a positive number.")
        if not isinstance(S1, Number) or isinstance(S1, complex) or S1 <= 0:
            raise ValueError("The input S1 must be a positive number.")
        if not isinstance(L, Number) or isinstance(L, complex) or L <= 0:
            raise ValueError("The input L must be a positive number.")
        if S0 > S1:
            raise ValueError("S0 must be strictly smaller than S1.")
        if S0 == S1:
            raise ValueError(
                "For a conical horn S0 must be strictly smaller than S1."
                "If S0 == S1, use :fun:`create_transmission_line` instead.",
            )

        r0 = np.sqrt(S0 / np.pi)
        r1 = np.sqrt(S1 / np.pi)

        d0 = 2 * r0
        d1 = 2 * r1

        a = r0 * (2 * L) / (d1 - d0)
        b = a + L

        Omega0 = S0 / a**2
        Omega1 = S1 / b**2

        if not np.isclose(Omega0, Omega1, atol=1e-15):
            raise ValueError(
                f"S0/a² is not equal to S1/b²: {Omega0} != {Omega1}."
                "This should never happen.",
            )

        Omega = Omega0  # or Omega1, they are equal

        return Omega, a, b

    @staticmethod
    def create_conical_horn(
        S0: Number,
        S1: Number,
        L: Number,
        k: Number | FrequencyData,
        medium_impedance: Number | FrequencyData = reference_air_impedance,
        propagation_direction: str = "forwards",
    ) -> TransmissionMatrix:
        r"""Create a transmission matrix representing a conical horn.

        The transmission matrix is calculated based on the starting point
        :math:`a`, end point :math:`b`, area constant :math:`\Omega`,
        wave number :math:`k`, and medium impedance :math:`Z_0`
        following Equation (5-18) of Reference [1]_:

        .. math::
            T = \begin{bmatrix}
                \frac{b}{a}cos(kl)-\frac{1}{ka}sin(kl) &
                \frac{jZ_0}{ab\Omega} \sin{kl} \\
                \frac{j\Omega}{k^2Z_0}\left( (1 + k^2ab) \sin{kl} -
                kl \cos{kl} \right) & \frac{a}{b}cos(kl)-\frac{1}{kb}sin(kl)
                \end{bmatrix}

        Parameters
        ----------
        S0 : float
            The surface area of the horn's narrow end.
        S1 : float
            The surface area of the horn's wide end.
        L : float
            The distance between the narrow and wide end of the horn,
            i.e. the horn's length.
        k : FrequencyData, scalar
            Wave number.
        medium_impedance : FrequencyData, scalar
            The impedance of the medium filling the horn. Defautl is
            ``pyfar.constants.reference_air_impedance``
        propagation_direction: str = {'forwards', 'backwards'}
            Defines the direction of sound propagation through the horn,
            where ``'forwards'`` means from narrow to wide end
            and ``'backwards'`` means from wide to narrow end.
            Default is ``'forwards'``.

        Returns
        -------
        pf.TransmissionMatrix
            Transmission matrix for the conical horn section.

        Example
        -------
        Arbitrarily sized lossless conical horn section.

        .. plot::

            >>> import pyfar as pf
            >>> import numpy as np
            >>> # Horn parameters
            >>> S0 = 0.003
            >>> S1 = 0.005
            >>> L = 0.3
            >>> frequencies = np.linspace(20, 20e3, 1000)
            >>> omega = 2 * np.pi * frequencies
            >>> k = pf.FrequencyData(
            >>>    omega / pf.constants.reference_speed_of_sound, frequencies)
            >>> Z0 = pf.constants.reference_air_impedance
            >>> # Create the transmission matrix
            >>> T = pf.TransmissionMatrix.create_conical_horn(
            >>>    S0, S1, L, k, Z0, 'backwards')
            >>> # Plot the transmission matrix
            >>> pf.plot.freq(T.input_impedance(np.inf))

        """
        if not (isinstance(k, FrequencyData) or isinstance(k, Number)):
            raise TypeError(
                "The wave number k must be a float, complex,"
                "or FrequencyData object.",
            )
        elif isinstance(k, FrequencyData):
            frequencies = k.frequencies
            k = k.freq
        else:
            if isinstance(k, complex):
                k_re = k.real
            else:
                k_re = k
            frequencies = (k_re * reference_speed_of_sound) / (2 * np.pi)
        if isinstance(medium_impedance, FrequencyData):
            if not np.allclose(
                medium_impedance.frequencies,
                frequencies,
                atol=1e-15,
            ):
                raise ValueError(
                    "The frequencies of characteristic_impedance must"
                    "match those of k.",
                )
            else:
                medium_impedance = medium_impedance.freq
        elif not isinstance(medium_impedance, Number):
            raise TypeError(
                "The input medium_impedance must be a number"
                "or a FrequencyData object.",
            )
        if not isinstance(propagation_direction, str):
            raise TypeError(
                "The input propagation_direction must be a string"
                "with value 'forwards' or 'backwards'.",
            )
        if propagation_direction not in ("forwards", "backwards"):
            raise ValueError(
                "The string propagation_direction must either be"
                "'forwards' or 'backwards'.",
            )

        if propagation_direction == "backwards":
            Omega, a, b = (
                TransmissionMatrix._calculate_horn_geometry_parameters(
                    S0,
                    S1,
                    L,
                )
            )  # Error handling inside the helper function
        else:
            Omega, b, a = (
                TransmissionMatrix._calculate_horn_geometry_parameters(
                    S0,
                    S1,
                    L,
                )
            )  # Error handling inside the helper function
            L = -1 * L

        A = b / a * np.cos(k * L) - 1 / (k * a) * np.sin(k * L)
        B = 1j * medium_impedance / (a * b * Omega) * np.sin(k * L)
        C = (
            1j
            * Omega
            / (k * k * medium_impedance)
            * ((1 + k * k * a * b) * np.sin(k * L) - k * L * np.cos(k * L))
        )
        D = a / b * np.cos(k * L) + 1 / (k * b) * np.sin(k * L)

        return TransmissionMatrix.from_abcd(A, B, C, D, frequencies)

    def __repr__(self):
        """String representation of TransmissionMatrix class."""
        repr_string = (
            f"TransmissionMatrix:\n"
            f"{self.abcd_cshape} channels per matrix entry "
            f"with {self.n_bins} frequencies\n"
            f"Total number of {self.cshape} channels"
        )

        return repr_string

    @classmethod
    def _decode(cls, obj_dict):
        """Decode object based on its respective `_encode` counterpart."""
        obj = cls(
            obj_dict["_data"], obj_dict["_frequencies"], obj_dict["_comment"]
        )
        obj.__dict__.update(obj_dict)
        return obj

    def is_indexable(self) -> bool:
        """Returns true if ABCD-entries have more than one channel and are
        therefore indexable.
        """
        return len(self.cshape) > 2

    def __getitem__(self, key):
        """Get copied slice of the TransmissionMatrix at key.

        Note, that slicing ABCD or frequency dimensions is not possible.
        """

        if not self.is_indexable():
            raise IndexError(
                "Object is not indexable, since ABCD-entries "
                "only have a single channel"
            )

        # Add three empty slices at the end to always get all data contained
        # in frequency and T-Matrix dimensions (last three dimensions).
        if hasattr(key, "__iter__"):
            key = (*key, slice(None), slice(None), slice(None))

        # try indexing and raise verbose errors if it fails
        try:
            data = self._data[key]
        except IndexError as error:
            if "too many indices for array" in str(error):
                raise IndexError(
                    (
                        f"Indexed dimensions must not exceed the ABCD "
                        f"channel dimension (abcd_cdim), which is "
                        f"{len(self.abcd_cshape)}"
                    )
                ) from error
            else:
                raise error

        return TransmissionMatrix.from_tmatrix(
            data, frequencies=self.frequencies, comment=self.comment
        )

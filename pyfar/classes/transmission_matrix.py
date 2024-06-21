"""
The following introduces the :py:func:`TransmissionMatrix class
<pyfar.classes.transmission_matrix.TransmissionMatrix>` available in pyfar.

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
    """ Class representing a transmission matrix

    This implementation is based on a paper by Lampton [#]_ and uses the ABCD-
    representation. A single T-matrix is a (2x2)-matrix of the form:
        [A  B]
        [C  D]
    This class represents frequency-dependent matrix of a multi-dimensional form,
    i.e. A, B, C and D can be matrices as well. For this purpose, it is derived
    from the :py:class:`~FrequencyData` class. In the easiest case, each matrix entry
    (A,B,C,D) is a vector depending on frequency. However, additional dimensions
    can be used to represent additional variables (e.g. multiple layouts of an
    electrical circuit).

    Notes
    -----
    This class is derived from :py:class:`~FrequencyData` but has a special constraint:
    The frequency data must match a shape like [..., 2, 2, N]. The last axis refers to
    the frequency, and the two axes before to the ABCD-Matrix (which is a 2x2 matrix).
    For example, `obj.freq[...,0,0,:]` returns the data related to the A-entry.

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
            In contrast to the :py:class:`~FrequencyData` class, data must have a shape
            of the form (..., 2, 2, N), e.g. (2, 2, 1024) or (3, 2, 2, 1024).
            In those examples 1024 refers to the number of frequency bins and
            the two dimensions before that to the (2x2) ABCD matrices.
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
        """Cascades two systems (T-matrices) by doing a matrix multiplication:

        See Equation (2-3) in [1]: T = T1 * T2

        Note: An easier approach especially to cascade multiple systems is using the @-operator, e.g. :
            T = T1 @ T2 @ T3 ...
        """
        return self @ t_matrix

    @property
    def abcd_axes(self):
        """The indices of the axes referring to the transmission matrix with respect to the full data set (including frequency-axis)"""
        return [-3, -2]

    @property
    def abcd_caxes(self):
        """The indices of the axes referring to the transmission matrix with respect to channel-related data set (excluding frequency-axis)"""
        return [-2, -1]

    @property
    def abcd_cshape(self):
        """
        The channel shape of the transmission matrix entries (A, B, C, D).
        This is the same as 'cshape' without the last two elements.
        """
        return self.cshape[:-2]

    @property
    def A(self) -> FrequencyData:
        """Returns the (potentially multi-dimensional) A entry of the t-matrix."""
        return self[..., 0, 0, :]

    @property
    def B(self) -> FrequencyData:
        """Returns the (potentially multi-dimensional) B entry of the t-matrix."""
        return self[..., 0, 1, :]

    @property
    def C(self) -> FrequencyData:
        """Returns the (potentially multi-dimensional) C entry of the t-matrix."""
        return self[..., 1, 0, :]

    @property
    def D(self) -> FrequencyData:
        """Returns the (potentially multi-dimensional) D entry of the t-matrix."""
        return self[..., 1, 1, :]


    def input_impedance(self, Zl: complex | FrequencyData):
        """Calculates the input impedance given the load impedance Zl at the output.

        See Equation (2-6) in reference [1]

                    o---xxxxxxxxx---o
                        x       x   |
            Zin-->      x       x   Zl
                        x       x   |
                    o---xxxxxxxxx---o
        """
        if np.shape(self.data[0, 0]) != np.shape(Zl):
            raise ValueError("'Zl' must match the dimensions of the matrix entries,"
                             "i.e. np.shape( self.freq[0,0] ) = np.shape(Zl)")
        return (self.A * Zl + self.B) / (self.C * Zl + self.D)

    def output_impedance(self, Zl: complex | FrequencyData):
        """Calculates the output impedance given the load impedance Zl at the input.

        See Equation (2-6) in reference [1]

                    o---xxxxxxxxx---o
                    |   x       x
                    Zl  x       x      <--Zout
                    |   x       x
                    o---xxxxxxxxx---o
        """
        if np.shape(self.data[0, 0]) != np.shape(Zl):
            raise ValueError("'Zl' must match the dimensions of the matrix entries,"
                             "i.e. np.shape( self.freq[0,0] ) = np.shape(Zl)")
        return (self.D * Zl + self.B) / (self.C * Zl + self.A)

    def transfer_function_quantity1(self, Zl: complex | FrequencyData):
        """
        Returns the transfer function of the first quantity (output/input) given the load at the output Zl
        The first quantity usually refers to voltage, force, pressure, etc.
        See Equation (2-1) in [1]: Defined as e2/e1 using i2 = e2/Zl
        """
        return 1 / (self.A + self.B / Zl)

    def transfer_function_quantity2(self, Zl: complex | FrequencyData):
        """
        Returns the transfer function of the second quantity (output/input) given the load at the output Zl
        The second quantity usually refers to some flow or movement, e.g. current, velocity,...
        See Equation (2-1) in [1]: Defined as i2/i1 using e2 = i2*Zl
        """
        return 1 / (self.C * Zl + self.D)

    # def transfer_function_quantity1_to_quantity2(self, Zl):
    #     pass
    # def transfer_function_quantity2_to_quantity1(self, Zl):
    #     pass


# TODO: Add static create functions for special matrices Eq. (2-8) to (2-9)

if __name__ == "__main__":
    tmat = TransmissionMatrix([1, 1], [2, 2], [3, 3], [4, 4], [100, 200])
    tmat.freq

r"""
pyfar implements five normalization [1]_ that can be applied to spectra. The
normalizations are available from :py:func:`~pyfar.dsp.fft.normalization`.
Note that the time signals do not change regardless of the normalization.


amplitude spectrum

.. math::
        \overline{X}(k) = \frac{1}{N} X(k)


power spectrum

.. math::
        \overline{\overline{X}}(k) = \lvert \overline{X}(k) \lvert ^2 =
        \frac{1}{N^2} \lvert X(k) \lvert ^2


power spectral density (PSD) or power desity spectrum

.. math::
        \overline{\overline{\underline{X}}}(k) = \frac{N}{f_s}
        \overline{\overline{X}}(k) = \frac{N}{f_s} \lvert \overline{X}(k)
        \lvert ^2 = \frac{1}{f_s N} \lvert  X(k) \lvert ^2


RMS spectrum

.. math::
        \overline{X}_{RMS}(k) = \left\{\begin{array}{ll}
        \frac{1}{\sqrt{2}} \overline{X}_{\text{SS}}(k) \quad \forall 0<k<
        \frac{N}{2}\\ \quad \; \overline{X}_{\text{SS}}(k) \quad \forall k=0,
        k= \frac{N}{2} \end{array}\right. .


References
----------
.. [1]  J. Ahrens, C. Andersson, P. Höstmad, and W. Kropp, “Tutorial on
        Scaling of the Discrete Fourier Transform and the Implied Physical
        Units of the Spectra of Time-Discrete Signals,” Vienna, Austria,
        May 2020, p. e-Brief 600.


See :py:mod:`~pyfar.dsp.fft` for a complete documentation.
"""

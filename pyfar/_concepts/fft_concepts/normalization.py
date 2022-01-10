r"""
Pyfar implements five normalizations [1]_ that can be applied to spectra. The normalizations are implictly used by the :py:class:`~pyfar.Signal` class and are available from :py:func:`~pyfar.dsp.fft.normalization`.
Note that the **time signals do not change** regardless of the normalization, instead the normalizations indicate how to **interpret the spectra**.

In order to illustrate the meaning of the normalizations in the following section, [1]_ is summarized and the consequences are discussed with respect to (arithmetic) operations.
Three signals are used for illustration purposes, all with N = 44100 number of samples and a length of 1s:

1. A (finite) impulse response signal, represented by a band pass signal (which e.g. could be a measured loudspeaker tranfer function, room impulse response etc.). In terms of signal types, it is considered as an *energy signal*.
2. A sine signal with an amplitude of 1, which represents a discrete tone (*power*) signal (where, e.g., a snippet was recorded of).
3. A white noise signal with an rms value of :math:`1/\sqrt{2}`, which represents a broadband stochastic (*power*) signal (where, e.g., a snippet was recorded of).



Considering all the signals as finite energy signals (which they are in a strict sense due to their finite representation on disk) is in not meaningful here, though their spectra look comparable. This is illustrated by comparing their energies :math:`W`, defined as

.. math::
        W = \sum_{n=0}^{N-1} s^2(n)

Spectra look similar: but: totaly different energy!
So considering strictly concept of power and energy, but: acoutstics, meaning.

The ``none`` normalization refers to the definition of the discrete Fourier spectrum

.. math::
        X(k) = \sum_{n=0}^{N-1} x(n) e^{-i 2 \pi \frac{k n}{N}}


unitary


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

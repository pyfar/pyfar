r"""
Pyfar implements five normalizations [1]_ that can be applied to spectra. The normalizations are implictly used by the :py:class:`~pyfar.Signal` class and are available from :py:func:`~pyfar.dsp.fft.normalization`. This means that, for a Signal object ``signal``, the spectrum data as ``signal.freq`` are calculated depending on the normalization given by ``signal.fft_norm``. **The time signals do not change regardless of the normalization.**

Example Signals
---------------

In order to illustrate the meaning of the normalizations in the following section, [1]_ is summarized and the consequences are discussed with respect to (arithmetic) operations.
Three signals are used for illustration purposes, all with sampling rate of *N = 44100 number of samples* and a *length of 1s*:

#. An impulse response (IR) signal represented by a band pass filter with a magnitude of :math:`1 \text{V}` in the pass band (which, e.g., could be the output of a microphone for a measured loudspeaker transfer function, a room impulse response ...). In terms of signal types, it is considered as an energy signal.
#. A sine signal with an amplitude of :math:`1 \text{V}`, which represents a discrete tone (power) signal (where, e.g., a snippet was recorded of).
#. A white noise signal with an rms value of :math:`1/\sqrt{2} \text{V}`, which represents a broadband stochastic (power) signal (where, e.g., a snippet was recorded of).
#. A FIR filter, represented by an energy preserving octave filter.

Note that the implied unit is :math:`1` for the FIR filter, not :math:`\text{V}`.

|example signals|

Note that a prefix of 10 was used for the dB calculations for the normalizations ``'power'`` and ``'psd'``.


Considering all the signals as finite energy signals (which they are in a strict sense due to their finite representation on disk) is in not meaningful here, though their spectra look comparable. This is illustrated by comparing their energies :math:`W`, defined as

.. math::
        W = \sum_{n=0}^{N-1} s^2(n)

Spectra look similar: but: totaly different energy!
So considering strictly concept of power and energy, but: acoutstics, meaning.

.. list-table::
   :header-rows: 1

   * - Norm
     - Equation
     - Explanation
   * - ``'none'``
     - :math:`X(k) = \sum_{n=0}^{N-1} x(n) e^{-i 2 \pi \frac{k n}{N}}``
     - Definition of the discrete Fourier transform ENERGY PRESERVING, NO CHANGE WITH PADDED ZEROS
   * - ``'unitary'``
     - :math:`X_{\text{SS}}(k) = \left\{\begin{array}{ll} X(k) & \forall k=0, k=\frac{N}{2}\\ 2 X(k) & \forall 0<k< \frac{N}{2} \end{array}\right.`
     -
   * - ``'amplitude'``
     - :math:`\overline{X}_{\text{SS}}(k) = \frac{1}{N} X_{\text{SS}}(k)`
     -
   * - ``'rms'``
     - :math:`\overline{X}_{RMS}(k) = \left\{\begin{array}{ll} \frac{1}{\sqrt{2}} \overline{X}_{\text{SS}}(k) = \frac{1}{\sqrt{2} N} X_{\text{SS}}(k) & \forall 0<k< \frac{N}{2}\\ \quad \overline{X}_{\text{SS}}(k) = \frac{1}{N} X_{\text{SS}}(k)  & \forall k=0, k=\frac{N}{2} \end{array}\right.`
     -
   * - ``'power'``
     - :math:`\overline{\overline{X}}(k) = \lvert \overline{X}(k) \lvert ^2 = \frac{1}{N^2} \lvert X(k) \lvert ^2`
     - = RMS in dB
   * - ``'psd'``
     - :math:`\overline{\overline{\underline{X}}}(k) = \frac{N}{f_s} \overline{\overline{X}}(k) = \frac{N}{f_s} \lvert \overline{X}(k) \lvert ^2 = \frac{1}{f_s N} \lvert  X(k) \lvert ^2`
     -

Note that, in contrast to _[1], all SignalSignals: single sided spectrum, for normalization not necessarily


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
        \frac{N}{2}\\ \overline{X}_{\text{SS}}(k) \quad \forall k=0,
        k = \frac{N}{2} \end{array}\right.


References
----------
.. [1]  J. Ahrens, C. Andersson, P. Höstmad, and W. Kropp, “Tutorial on
        Scaling of the Discrete Fourier Transform and the Implied Physical
        Units of the Spectra of Time-Discrete Signals,” Vienna, Austria,
        May 2020, p. e-Brief 600.


See :py:mod:`~pyfar.dsp.fft` for a complete documentation.


.. |example signals| image:: resources/fft_norms_examples.png
   :width: 100%
   :alt: Examples signals to illustrate the FFT-normalizations
"""

r"""
Pyfar implements five normalizations [1]_ that can be applied to spectra. The
normalizations are implicitly used by the
:py:class:`~pyfar.classes.audio.Signal`
class and are available from :py:func:`~pyfar.dsp.fft.normalization`. This
means that, for a Signal object ``signal``, ``signal.freq`` is calculated
depending on the normalization given by ``signal.fft_norm``. **The time signals
do not change regardless of the normalization.**

In order to illustrate the meaning of the normalizations, [1]_ is summarized
and the consequences are discussed with respect to (arithmetic) operations.

Definitions
-----------

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Norm
     - Equation
   * - ``'none'``
     - :math:`X(k) = \sum_{n=0}^{N-1} x(n) e^{-i 2 \pi \frac{k n}{N}}``
   * - ``'unitary'``
     - :math:`X_{\text{SS}}(k) = \left\{\begin{array}{ll} X(k) & \forall k=0,
       k=\frac{N}{2}\\ 2 X(k) & \forall 0<k< \frac{N}{2} \end{array}\right.`
   * - ``'amplitude'``
     - :math:`\overline{X}_{\text{SS}}(k) = \frac{1}{N} X_{\text{SS}}(k)`
   * - ``'rms'``
     - :math:`\overline{X}_{RMS}(k) = \left\{\begin{array}{ll}
       \frac{1}{\sqrt{2}} \overline{X}_{\text{SS}}(k) & \forall 0<k<
       \frac{N}{2}\\ \quad \overline{X}_{\text{SS}}(k) & \forall k=0,
       k=\frac{N}{2} \end{array}\right.`
   * - ``'power'``
     - :math:`\overline{\overline{X}}_{\text{SS}}(k) = \lvert
       \overline{X}_{\text{RMS}}(k) \lvert ^2`
   * - ``'psd'``
     - :math:`\overline{\overline{\underline{X}}}_{\text{SS}}(k) =
       \frac{N}{f_s} \overline{\overline{X}}_{\text{SS}}(k) = \frac{N}{f_s}
       \lvert \overline{X}_{\text{RMS}}(k) \lvert ^2`


Note that all pyfar signals are real-valued, leading to single-sided spectra.
So there are small differences in the definitions compared to the formulas
written in [1]_.

Explanations
------------

* ``'none'``:
        The definition of the discrete Fourier transform as used in
        pyfar. It represents the signal's energy, i.e., zeropadding does not
        change spectrum. Accordingly, this norm is to be used for energy
        signals such as impulse responses. For power signals, the magnitudes
        depend on the number of samples ("longer recording = more energy"),
        therefore a different normalization is appropriate.
* ``'unitary'``:
        In pyfar, all spectra are single-sided. For power signals,
        this conversion requires to multiply the outcome of the discrete
        Fourier transform with a factor of 2 in order to represent power
        related measures correctly (e.g., the amplitude or RMS-value, see
        below).
* ``'amplitude'``:
        This normalization considers the dependence of the spectral
        magnitudes on the number of samples, which is appropriate for discrete
        tones as the resulting magnitudes in the spectrum can be interpreted as
        amplitudes.
* ``'rms'``:
        The crest factor of sine waves of :math:`\sqrt{2}` is considered
        to represent RMS-values (a power quantity) resulting in a difference of
        -3 dB compared to ``'amplitude'``.
* ``'power'``:
        In a dB representation, it equals the ``'rms'`` normalization
        describing a power quantity. For stochastic broadband signals, this
        results in a dependence of the magnitude on the sampling rate as these
        are defined by a constant power density (see ``'psd'``).
* ``'psd'``:
        Using this normalization, signals are represented as
        *power densities* (e.g. in V²/Hz), leading to a meaningful
        representation for broadband stochastic signals but not for discrete
        tones or impulse responses.

.. list-table:: **Overview on appropriate normalizations**
   :widths: 20 35 45
   :header-rows: 1

   * - Signal type
     - Variation
     - Normalization
   * - Energy
     - Impulse responses
     - ``'none'``
   * - Power
     - Discrete tones
     - ``'amplitude'``, ``'rms'``, ``'power'``
   * - Power
     - Broadband stochastic signals
     - ``'psd'``


Examples
--------

Three signals are used for illustration purposes, all with sampling rate of
1000 samples and a sampling rate of 10 kHz:

#. An impulse signal (one at :math:`t=0` and zero otherwise) with a constant
   spectrum. It is an energy signal, so the appropriate normalization is
   ``'none'``.
#. A FIR filter. This energy preserving octave filter represents a
   general impulse response / transfer function with finite energy (e.g., a
   measured loudspeaker transfer function, a room impulse response, HRTF ...) It
   is an energy signal, so the appropriate normalization is ``'none'``.
#. A sine signal with an amplitude of :math:`1 \text{V}`. It represents a discrete tone
   of which a snippet was recorded. Accordingly, it possess a finite power but
   infinite energy, so it is a power signal with appriate normalizations
   ``'amplitude'``, ``'rms'`` or ``'power'``.
#. A white noise signal with an RMS
   value of :math:`1/\sqrt{2} \text{V}`. It represents a broadband stochastic
   signal of which a snippet was recorded of. Accordingly, the appropriate
   normalization is ``'psd'``.

Note that the implied units differ and a prefix of 10 is used for the dB
calculations of the normalizations ``'power'`` and ``'psd'``.

|examples|


* The ``'none'`` normalization gives expected results for the impulse and FIR
  Filter, but leads to a magnitude of number of samples/2 for the sine signal
  (1000/2, 60-6 dB). As illustrated, other normalizations than ``'none'`` are
  not meaningful for the IRs.
* For the sine signal, ``'unitary'`` normalization considers the factor 2 due
  to the single-side spectrum (+6 dB compared to ``'none'``). The
  ``'amplitude'`` normalization considers the number of samples, so the
  amplitude of 1 V (= 0 dB) is represented in the spectrum. Accordingly, the
  magnitude of the sine is -3 dBV with ``'rms'`` and ``'power'`` normalization.
* With ``'psd'`` normalization, the sine's magnitude is reduced by a factor of
  number of samples / sampling rate (1/10, -10 dB). As discussed above, this
  normalization is only meaningful for the noise, as is represents a spectral
  density.

For further details, especially on the the background for the power
normalizations, it is referred to [1]_.

References
----------
.. [1]  J. Ahrens, C. Andersson, P. Höstmad, and W. Kropp, “Tutorial on Scaling
        of the Discrete Fourier Transform and the Implied Physical Units of the
        Spectra of Time-Discrete Signals,” Vienna, Austria, May 2020, p.
        e-Brief 600.


.. |examples| image:: resources/fft_norms_examples.png
   :width: 100%
   :alt: FFT Normalization Examples
"""

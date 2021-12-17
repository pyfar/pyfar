r"""
Definition
----------
The discrete Fourier spectrum of an arbitrary, but band-limited signal
:math:`x(n)` is defined as

.. math::
        X(\mu) = \sum_{n=0}^{N-1} x(n) e^{-i 2 \pi \frac{\mu n}{N}}

using a negative sign convention in the transform kernel
:math:`\kappa(\mu, n) = e^{-i 2 \pi \mu \frac{n}{N}}`.
Analogously, the discrete inverse Fourier transform is implemented as

.. math::
        x(n) = \frac{1}{N} \sum_{\mu=0}^{N-1} X(\mu) e^{i2\pi\frac{\mu n}{N}}

Pyfar uses a DFT implementation for purely real-valued time signals resulting
in Fourier spectra with complex conjugate symmetry for negative and
positive frequencies :math:`X(\mu) = X(-\mu)^*`. As a result,
the left-hand side of the spectrum is discarded, yielding
:math:`X_R(\mu) = X(\mu) \mbox{ }\forall 0 \le \mu \le N/2`. Complex valued
time signals can be implemented, if required.

Normalization [1]_
------------------
pyfar implements five normalization that can be applied to spectra. The
normalizations are available from :py:func:`~pyfar.dsp.fft.normalization`.
Note that the time signals do not change regardless of the normalization.

Energy Signals
==============

For energy signals with finite energy,
such as impulse responses, no normalization is required, that is
the spectrum of a energy signal is equivalent to the right-hand spectrum
of a real-valued time signal defined above. The corresponding normalization is
``'none'``.

Power Signals
=============

For power signals however, which possess a finite power but infinite energy,
a normalization for the time interval in which the signal is sampled, is
chosen. In order for Parseval's theorem to remain valid, the single sided
needs to be multiplied by a factor of 2, compensating for the discarded part
of the spectrum (cf. [1]_, Eq. 8). The coresponding normalization is
``'unitary'``. Additional normalizations can be applied to further scale the
spectrum, e.g., according to the RMS value.

.. plot::

    >>> import numpy as np
    >>> from pyfar.dsp import fft
    >>> import matplotlib.pyplot as plt
    >>> # properties
    >>> fft_normalization = "rms"
    >>> n_samples = 1024
    >>> sampling_rate = 48e3
    >>> frequency = 100
    >>> times = np.linspace(0, 10, n_samples)
    >>> freqs = fft.rfftfreq(n_samples, 48e3)
    >>> # generate data
    >>> sine = np.sin(times * 2*np.pi * frequency)
    >>> spec = fft.rfft(sine, n_samples, sampling_rate, fft_normalization)
    >>> # plot time and frequency data
    >>> plt.subplot(1, 2, 1)
    >>> plt.plot(times, sine)
    >>> ax = plt.gca()
    >>> ax.set_xlabel('Time in s')
    >>> plt.subplot(1, 2, 2)
    >>> plt.plot(freqs, np.abs(spec))
    >>> ax = plt.gca()
    >>> ax.set_xlabel('Frequency in Hz')
    >>> plt.show()


References
----------
.. [1]  J. Ahrens, C. Andersson, P. Höstmad, and W. Kropp, “Tutorial on
        Scaling of the Discrete Fourier Transform and the Implied Physical
        Units of the Spectra of Time-Discrete Signals,” Vienna, Austria,
        May 2020, p. e-Brief 600.


See :py:mod:`~pyfar.dsp.fft` for a complete documentation.
"""

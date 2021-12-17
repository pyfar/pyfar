r"""
For **energy signals** with finite energy,
such as impulse responses, no normalization is required, that is
the spectrum of a energy signal is equivalent to the right-hand spectrum
of a real-valued time signal defined above. The corresponding normalization is
``'none'``.

For **power signals** however, which possess a finite power but infinite
energy, a normalization for the time interval in which the signal is sampled,
is chosen. In order for Parseval's theorem to remain valid, the single sided
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
"""

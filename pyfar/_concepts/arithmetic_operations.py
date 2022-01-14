"""
Arithmetic operations can be applied in the time and frequency domain and
are implemented in the methods ``add``, ``subtract``, ``multiply``, ``divide``,
and ``power``. For example, two :py:func:`~pyfar.classes.audio.Signal`,
:py:func:`~pyfar.classes.audio.TimeData`, or
:py:func:`~pyfar.classes.audio.FrequencyData` instances can be added in the
time domain by

>>> result = pyfar.classes.audio.add((signal_1, signal_2), 'time')

and in the frequency domain by

>>> result = pyfar.classes.audio.add((signal_1, signal_2), 'freq')

This also works with more than two instances and supports array likes and
scalar values, e.g.,

>>> result = pyfar.classes.audio.add((signal_1, 1), 'time')

In this case the scalar `1` is broadcasted, i.e., it is is added to every
sample of `signal` (or every bin in case of a frequency domain operation).

The operators ``+``, ``-``, ``*``, ``/``, and ``**`` are overloaded for
convenience. Note, however, that their behavior depends on the Audio object.
Frequency domain operations are applied for
:py:func:`~pyfar.classes.audio.Signal` and
:py:func:`~pyfar.classes.audio.FrequencyData` objects, i.e,

>>> result = signal1 + signal2

is equivalent to

>>> result = pyfar.classes.audio.add((signal1, signal2), 'freq')

Time domain operations are applied for
:py:func:`~pyfar.classes.audio.TimeData` objects, i.e.,

>>> result = time_data_1 + time_data_2

is equivalent to

>>> result = pyfar.classes.audio.add((time_data_1, time_data_2), 'time')

In addition to the arithmetic operations, the equality operator is overloaded
to allow comparisons

>>> signal_1 == signal_2

See :py:class:`~pyfar.classes.audio` for a complete documentation.


FFT Normalizations
------------------
The arithmetic operations are implemented in a way that only physically meaningful arithmetic operations are allowed with respect to the FFT normalizations of the signals. These rules are motivated by the fact that the normalizations correspond to specific types of signals (e.g., energy signals, discrete tone signals, stochastic broadband signals). While addition and subtraction are independent of being operated in time or frequency domain, this is not necessarily the case for multiplication and division. Nevertheless, **the same rules apply for both time and frequency domain operations** for convenience:

Addition, subtraction and multiplication
****************************************

* Either: one signal has ``fft_norm`` ``'none'`` , the results gets the other normalization.
* Or: both have the same ``fft_norm``, the results gets the same normalization.
* Other combinations raise an error.

Division
********

* Either: the denominator has the ``fft_norm`` ``'none'``, the result gets the ``fft_norm`` of the numerator.
* Or: both have the same fft_norm, the results gets the fft_norm ``'none'``.
* Other combinations raise an error.
"""

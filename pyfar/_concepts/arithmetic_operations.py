"""
Arithmetic operations can be applied in the time and frequency domain and
are implemented in the methods :py:func:`~pyfar.classes.audio.add`,
:py:func:`~pyfar.classes.audio.subtract`,
:py:func:`~pyfar.classes.audio.multiply`,
:py:func:`~pyfar.classes.audio.divide` and
:py:func:`~pyfar.classes.audio.power`. For example, two
:py:func:`~pyfar.classes.audio.Signal`,
:py:func:`~pyfar.classes.audio.TimeData`, or
:py:func:`~pyfar.classes.audio.FrequencyData` instances can be added in the
time domain by

>>> result = pf.add((signal_1, signal_2), 'time')

and in the frequency domain by

>>> result = pf.add((signal_1, signal_2), 'freq')

**Note** that frequency domain operations are performed on the raw spectrum
``signal.freq_raw``.

Arithmetic operations also work with more than two instances and support
array likes and scalar values, e.g.,

>>> result = pf.add((signal_1, 1), 'time')

In this case the scalar `1` is broadcasted, i.e., it is is added to every
sample of `signal` (or every bin in case of a frequency domain operation).
The shape of arrays need to match the ``cshape`` of the resulting audio object,
e.g.,

>>> x = np.arange(2 * 3 * 4).reshape((2, 3, 4))
>>> y = pf.Signal(np.ones((2, 3, 4, 10)), 44100)
>>> z = pf.add((x, y))

or are broadcasted, e.g.,

>>> x = np.arange(3 * 4).reshape((3, 4))
>>> y = pf.Signal(np.ones((2, 3, 4, 10)), 44100)
>>> z = pf.add((x, y))

The operators ``+``, ``-``, ``*``, ``/``, and ``**`` are overloaded for
convenience. Note, however, that their behavior depends on the Audio object.
Frequency domain operations are applied for
:py:func:`~pyfar.classes.audio.Signal` and
:py:func:`~pyfar.classes.audio.FrequencyData` objects, i.e,

>>> result = signal1 + signal2

is equivalent to

>>> result = pf.add((signal1, signal2), 'freq')

and time domain operations are applied for
:py:func:`~pyfar.classes.audio.TimeData` objects, i.e.,

>>> result = time_data_1 + time_data_2

is equivalent to

>>> result = pf.add((time_data_1, time_data_2), 'time')

In addition to the arithmetic operations, the equality operator is overloaded
to allow comparisons

>>> signal_1 == signal_2

See :py:class:`audio classes <pyfar.classes.audio>` for a complete
documentation.


FFT Normalizations
------------------
The arithmetic operations are implemented in a way that only physically
meaningful operations are allowed with respect to the
:py:mod:`FFT normalization <pyfar._concepts.fft>`. These rules are motivated by
the fact that the normalizations correspond to specific types of signals (e.g.,
energy signals, pure tone signals, stochastic broadband signals). While
addition and subtraction are equivalent in the time and frequency domain,
this is not the case for multiplication and division. Nevertheless, **the same
rules apply regardless of the domain** for convenience:

Addition, subtraction and multiplication
****************************************

* If one signal has the FFT normalization ``'none'`` , the results gets the
  normalization of the other signal.
* If both signals have the same FFT normalization, the results gets the same
  normalization.
* Other combinations raise an error.

Division
********

* If the denominator signal has the FFT normalization ``'none'``, the result
  gets the normalization of the numerator signal.
* If both signals have the same FFT normalization, the results gets the
  normalization ``'none'``.
* Other combinations raise an error.
"""

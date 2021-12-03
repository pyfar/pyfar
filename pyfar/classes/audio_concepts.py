"""
Audio classes
-------------

The classes :py:func:`~pyfar.classes.audio.TimeData` and
:py:func:`~pyfar.classes.audio.FrequencyData` are intended to
store incomplete or non-equidistant audio data in the time and frequency
domain. The class :py:func:`~pyfar.classes.audio.Signal` can be used to store
equidistant and
complete audio data that can be converted between the time and frequency
domain by means of the Fourier transform.

Arithmethic operations
----------------------

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

"""

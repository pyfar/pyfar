"""
Audio data are at the core or pyfar and three classes can be used for storing,
processing, and visualizing such data

- The :py:func:`~pyfar.classes.audio.Signal` class can be used to store
  equidistant and complete signals that can be converted between the time and
  frequency domain.
- The :py:func:`~pyfar.classes.audio.TimeData` and
  :py:func:`~pyfar.classes.audio.FrequencyData` classes are intended for
  incomplete or non-equidistant audio data in the time and frequency domain
  that can *not* be converted between the time and frequency domain.

All three classes provide methods for accessing the data and useful meta data
such as the sampling rate or number of channels, and almost all functions in
pyfar take audio classes as their main input. See
:py:class:`audio classes <pyfar.classes.audio>` for a complete documentation.

**Data inside audio classes:** ``cshape`` **,** ``cdim`` **and** ``caxis``

All audio classes can store multi-dimensional data. For example 3 by 4
channel audio data with 128 samples of data per channel would be stored in an
array of shape 3 by 4 by 128, also written as ``(3, 4, 128)``. In audio signal
processing, operations are often performed on channels. To handle this
conveniently, pyfar defines the channel shape - or ``cshape`` in short - which
ignores how many samples or frequency bins are stored in an audio object. In
the above example the cshape would be ``(3, 4)`` and `not` (3, 4, 128).

In analogy, the channel axis - or ``caxis`` in short - refers to the dimension
or axis of the cshape. For example ``caxis = 0`` refers to the first dimension
of size 3, and ``caxis = -1`` refers to the last dimension of size 4 and `not`
to the last axis of the data array of size 128. Note that ``caxis`` can also be
provided as a tuple or list, e.g., ``caxis = (0, 1)`` refers to both channel
axes of size 3 and 4.

Lastly the channel dimension - or ``cdim`` in short - is the length of the
``cshape``.

**Signal Types**

The :py:func:`~pyfar.classes.audio.Signal` class distinguishes between two
kinds of signals

- *energy signals* are of finite length and energy, such as impulse responses.
- *power signals* are finite samples of signals with infinite length and
  energy, such as noise signals or sound textures.

The difference is important for

- plotting the frequency response of signals because different signal types
  require different :py:mod:`FFT normalizations <pyfar._concepts.fft>`.
- performing
  :py:mod:`arithmetic operations <pyfar._concepts.arithmetic_operations>`
  because not all signal types can be combined with each other.

"""

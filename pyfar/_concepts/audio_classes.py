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

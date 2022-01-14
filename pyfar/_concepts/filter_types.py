"""
:py:mod:`pyfar.dsp.filter` contains different filter types that are briefly
introduced in the following. The filters can be used to directly
filter a :py:class:`~pyfar.classes.audio.Signal` or can return a
:py:mod:`Filter object <pyfar.classes.filter>` described in
:py:mod:`Filter classes <pyfar._concepts.filter_classes>`.


High-pass, low-pass, band-pass, and band-stop filters
-----------------------------------------------------

These are the classic filters that are wrapped from ``scipy.signal`` and
available from :py:func:`~pyfar.dsp.filter.butterworth`,
:py:func:`~pyfar.dsp.filter.bessel`, :py:func:`~pyfar.dsp.filter.chebyshev1`,
:py:func:`~pyfar.dsp.filter.chebyshev2`, and
:py:func:`~pyfar.dsp.filter.elliptic`

|standard_filter|


Linkwitz-Riley cross-over
-----------------------------------------------------
The functions :py:func:`~pyfar.dsp.filter.crossover`, realized Linkwitz-Riley
cross-over filters that are often used in loudspeaker design.

|crossover|


Filter banks
-----------------------------------------------------

Filter banks are commonly used in audio and acoustics signal processing and
pyfar contains two types of filter banks:

- The :py:func:`~pyfar.dsp.filter.fractional_octave_bands` are often used for
  calculating room acoustic parameters
- The :py:func:`~pyfar.dsp.filter.reconstructing_fractional_octave_bands` can
  be used if a perfect reconstruction is required, e.g., in room acoustical
  simulations.

|filter_banks|


Parametric equalizer
-----------------------------------------------------
The functions :py:func:`~pyfar.dsp.filter.high_shelve`,
:py:func:`~pyfar.dsp.filter.low_shelve`, and :py:func:`~pyfar.dsp.filter.bell`
are more specific filter for digital audio signal processing and often used
in audio effects and loudspeaker or room compensation.

|eqs|


.. |standard_filter| image:: resources/filter_types_standard.png
   :width: 100%
   :alt: Standard filters contained in pyfar

.. |filter_banks| image:: resources/filter_types_filterbanks.png
   :width: 100%
   :alt: Filter banks contained in pyfar

.. |crossover| image:: resources/filter_types_crossover.png
   :width: 50%
   :alt: Cross-over contained in pyfar

.. |eqs| image:: resources/filter_types_parametric-eq.png
   :width: 50%
   :alt: Parametric equalizer contained in pyfar
"""

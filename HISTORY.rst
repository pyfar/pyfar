=======
History
=======

0.2.0 (2021-06-01)
------------------
* Add DSP functions: `zero_phase`, `time_window`, `linear_phase`, `pad_zeros`, `time_shift`, `minimum_phase`,
* Add DSP class `InterpolateSpectrum`
* Add reconstructing fractional octave filter bank
* Unified the `unit` parameter in the pyfar.dsp module to reduce duplicate code. Unit can now only be `samples` or `s` (seconds) but not `ms` or `mus` (milli, micro seconds)
* Bugfix for mis-matching filter slopes in `crossover` filter
* Refactored internal handling of filter functionality for filter classes
* Added functionality to save/read filter objects to/from disk
* Improved unit tests
* Improved documentation

0.1.0 (2021-04-11)
------------------
* First release on PyPI

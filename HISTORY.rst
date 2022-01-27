=======
History
=======

0.3.0 (2022-01-28)
------------------
* More reasonable handling of FFT normalizations in arithmetic operations (pyfar.readthedocs.io/en/latest/concepts/pyfar.arithmetic_operations.html). This is a major change and might break backwards compatability in some cases (although this is unlikely to happen, PR #253, #245, #235).
* The documentation now contains concepts behind pyfar in a separate section. This makes the concepts easier to find and understand and the documetation of the classes and functions cleaner (PR #251, #243).
* `pyfar.dsp`
  * Added `convolve` for convolution of signals in the time and frequency domain (PR #232)
  * Added `deconvolve` for frequency domain deconvolution with the optional regularization (PR #212)
  * functions in the `filter` module have more verbose names, e.g., 'butterworth' instead of 'butter'. Functions with old names will be deprecated in pyfar 0.5.0 (PR #248).
  * `time_window` can now return the window to make it easer to insepct windows and apply windows multiple times (PR #247)
  * the dB parameters in `spectrogram` were never working. They were thus removed and can be controlled in the plot function `pyfar.plot.spectrogram` instead (PR #258, #256).
* `pyfar.io`
  * added `read_audio` and `write_audio` to support more types of audio files (based on the `soundfile` package). The old functions `read_wav` and `write_wav` will be deprecated in pyfar 0.5.0 (PR #234)
  * `read_sofa` can now also load SOFA files of DataType 'TransferFunction' (e.g. GeneralTF) and uses the sofar package (sofar.readthedocs.io, PR #254, #240).
* `pyfar.plot`
  * Plots of the magnitude spectrum now use `10` as the new default `log_prefix` for calculating the level in dB for plotting Signals with the FFT normalizations 'psd' and 'power' (PR #260)
  * `custom_subplot` now returns axis handles (PR #237)
  * Frequency plots allow to show negative frequencies (PR #233)
* Filter classes (`pyfar.FilterFIR`, `pyfar.FilterIIR`, `pyfar.FilterSOS`)
  * Rename the property `shape` to `n_channels`. pyfar Filter objects to not support multi-dimensional layouts (PR #102)
  * Filter states can now be saved to allow block-wise processing (PR #102)
  * The `coefficients` can now be set. This allows to mimic time variant systems in block-wise processing (PR #252)
  * Improved documentation (PR #252)
* CI: Only test wheels to save time during testing (PR #236)
* Enhanced contributing guidelines (PR #239)

0.2.3 (2021-11-12)
------------------
* Fix broken install on Python 3.9
* Remove fft normalizations from FrequencyData

0.2.2 (2021-11-05)
------------------
* Removed dependency on pyfftw in favor of scipy.fft to support Python 3.9 and above

0.2.1 (2021-10-12)
------------------
* Bugfix for left and right hand side arithmetic operators

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

=======
History
=======

0.5.0 (2022-10-13)
------------------
* General

  * End support for Python 3.7 because it was deprecated in numpy functionality also used by pyfar (PR #350)
  * Deprecate `read_wav` and `write_wav` from the `pyfar.io` module in favor or `read_audio` and `write_audio` (PR# 310)
  * Deprecate the `get_nearst_*` functions from the `Coordinates` class in favor of `find_nearest_*` functions (PR #310)
  * Deprecate `linear_sweep` and `exponential_sweep` from the `pyfar.signals` module in favor or `linear_sweep_time` and `exponential_sweep_time` (PR #310)
  * Deprecate cryptic names in `pyfar.dsp.filter` module for more verbose names, e.g., `butter` was deprecated in favor of `butterworth` (PR #310)
  * Improved Documentation and bugfixes (PR #324, #354, #355)

* Audio classes (`Signal`, `TimeData`, and `FrequencyData`)

  * Added matrix multiplication to arithmetic operations (PR #277)
  * Improved broadcasting and documentation for arithmetic operations (PR #318)
  * The data type is now automatically derived from the input. The parameter `dtype` was removed and the class structure improved (PR #344)

* `pyfar.dsp`

  * Improved algorithm of `minimum_phase` for arbitrary impulse responses (PR #303)
  * Added `resample` function for sample rate conversions (PR #297, #321, #333)
  * Added `find_impulse_response_start` and `find_impulse_response_delay` to detect the time of arrival in impulse responses (PR # 203)
  * Added `normalize` function for time and frequency domain normalization (PR #323)
  * Added `energy`, `power`, and `rms` for computing energy measures in the time domain (PR #338)
  * Added `time_shift` function for applying linear and cyclic integer delays (PR #312)
  * Added `fractional_time_shift` function for applying linear and cyclic fractional delays (PR # 292)
  * Added `fractional_octave_smoothing` function (PR #297)
  * Added `decibel` function (PR #305, #322)
  * Added new mandatory parameter `freq_range` to `deconvolve` (PR #370)

* `pyfar.dsp.filter`

  * Added reconstructing auditory `GammatoneBands` filter bank (PR #327)

* `pyfar.signals`

  * Improved flexibility and broadcasting of parameters for `impulse` and `sine` signals (PR #313)

* `pyfar.io`

  * Added `read_comsol` and `read_comsol_header` to import data from COMSOL (PR #339)
  * Include updates incl. MP3 support from `soundfile v0.11.0 <https://python-soundfile.readthedocs.io/en/0.11.0/#news>`_ for `write_audio` and `read_audio` (PR #365)

* `pyfar.plot`

  * Time domain plots now always use seconds as the default unit. The previous default `'auto'` caused unexpected behavior by changing the unit of already existing plots depending on the lengths of the Signal that was plotted last (PR #308)

* Other

  * Test building the documentation using CI (PR #319, #348)
  * Fixed broken mybinder.org examples (PR #341)
  * Internal refactoring, documentation, and bug fixes (PR #326, #331, #352)

0.4.3 (2022-08-08)
------------------
* Make python-soundfile an optional requirement due to unsupported architectures. Note that without python-soundfile common audio file format are no longer supported via `pyfar.io` (PR #334, #340).
* Developer: Switch to CircleCI for continuous testing (PR #336).

0.4.2 (2022-05-20)
------------------
* Bugfix: Sweep functions marked for deprecation had no return value.

0.4.1 (2022-04-08)
------------------
* Bugfix: do not allow 'flat' shading parameter in 2D plot functions (PR #291)

0.4.0 (2022-03-02)
------------------
* `pyfar.plot`

  * The plot module was largely extended by 2D color coded versions of the former line plot functions: `time_2d`, `freq_2d`, `phase_2d`, `group_delay_2d`, `time_freq_2d`, `freq_phase_2d` and `freq_group_delay_2d`. New shortcuts for interactive plots were added to cycle between line and 2D plots and to toggle between vertical and horizontal orientation of 2D plots. (PR #198, #273, #276)
  * The `xscale` parameter was replaced by the more explicit `freq_scale` parameter in all plot functions. It will be removed in pyfar 0.6.0 (PR #282)

* `pyfar.filter`

  * Added cascaded shelving filters `low_shelve_cascade` and `high_shelve_cascade` used to generate filters with a user definable slope given in dB per octaves within a certain frequency region. (PR #284)

* `pyfar.Signal`

  * Added a `freq_raw` property, which is the frequency spectrum without normalization. It enables easy access and reduces complexity in internal computations. (PR #274)

0.3.0 (2022-01-28)
------------------
* More reasonable handling of FFT normalizations in `arithmetic operations <https://pyfar.readthedocs.io/en/latest/concepts/pyfar.arithmetic_operations.html>`_. This is a major change and might break backwards compatibility in some cases (although this is unlikely to happen, PR #253, #245, #235).
* The documentation now contains `concepts <https://pyfar.readthedocs.io/en/latest/concepts.html>`_ behind pyfar in a separate section. This makes the concepts easier to find and understand and the documentation of the classes and functions cleaner (PR #251, #243).

* `pyfar.dsp`

  * Added `convolve` for convolution of signals in the time and frequency domain (PR #232)
  * Added `deconvolve` for frequency domain deconvolution with the optional regularization (PR #212)
  * functions in the `filter` module have more verbose names, e.g., 'butterworth' instead of 'butter'. Functions with old names will be deprecated in pyfar 0.5.0 (PR #248).
  * `time_window` can now return the window to make it easier to inspect windows and apply windows multiple times (PR #247)
  * the dB parameters in `spectrogram` obsolete. They were thus removed and can be controlled in the plot function `pyfar.plot.spectrogram` instead (PR #258, #256).

* `pyfar.io`

  * `pyfar.io.read` and `pyfar.io.write` can now handle Python built in data types (PR #205)
  * added `read_audio` and `write_audio` to support more types of audio files (based on the `soundfile` package). The old functions `read_wav` and `write_wav` will be deprecated in pyfar 0.5.0 (PR #234)
  * `read_sofa` can now also load SOFA files of DataType 'TransferFunction' (e.g. GeneralTF) and uses the `sofar <https://sofar.readthedocs.io>`_ package (PR #254, #240).

* `pyfar.plot`

  * Plots of the magnitude spectrum now use ``10`` as the new default `log_prefix` for calculating the level in dB for plotting Signals with the FFT normalizations ``'psd'`` and ``'power'`` (PR #260)
  * Improved handling of colorbar in `pyfar.plot.spectrogram`. A speparate axis for the colorbar can be passed to the function. The function can return the axis of the colorbar. (PR #216)
  * `custom_subplot` now returns axis handles (PR #237)
  * Frequency plots allow to show negative frequencies (PR #233)

* Filter classes (`pyfar.FilterFIR`, `pyfar.FilterIIR`, `pyfar.FilterSOS`)

  * Rename the property `shape` to `n_channels`. pyfar Filter objects do not support multi-dimensional layouts (PR #102)
  * Filter states can now be saved to allow block-wise processing (PR #102)
  * The `coefficients` can now be set. This allows to mimic time variant systems in block-wise processing (PR #252)
  * Improved documentation (PR #252)

* Audio classes (`pyfar.Signal`, `pyfar.TimeData`, `pyfar.FrequencyData`)

  * Make arithmetic operations available as `pyfar.add`, `pyfar.subtract`, etc. (PR # 230)
  * Remove fft normalizations from FrequencyData (PR #225)

* `pyfar.Coordinates` and `pyfar.Orientations`

  * Renamed methods `pyfar.Coordinates.get_nearest_*` to `pyfar.Coordinates.find_nearest_*`. Old methods will be deprecated in pyfar 0.5.0 (PR #209)
  * The plots generated by `Coordinates.show` and `Orientations.show` now use the pyfar plot style (PR #169)

* `pyfar.signals`

  * renamed `pyfar.signals.linear_sweep` to `pyfar.signals.linear_sweep_time` and `pyfar.signals.exponential_sweep` to `pyfar.signals.exponential_sweep_time`. Old functions will be deprecated in pyfar 0.5.0 (PR # 201)

* CI: Only test wheels to save time during testing (PR #236)
* Enhanced contributing guidelines (PR #239)

0.2.3 (2021-11-12)
------------------
* Fix broken install on Python 3.9

0.2.2 (2021-11-05)
------------------
* Removed dependency on pyfftw in favor of scipy.fft to support Python 3.9 and above (PR #227)

0.2.1 (2021-10-12)
------------------
* Bugfix for left and right hand side arithmetic operators (PR #226)

0.2.0 (2021-06-01)
------------------
* `pyfar.dsp`

  * added `linear_phase` (PR #176)
  * added `minimum_phase` (PR #185)
  * added `zero_phase` (PR #175)
  * added `time_window` (PR #178)
  * added `pad_zeros` (PR #184)
  * added `time_shift` (PR #186)
  * added `InterpolateSpectrum` (PR #187)
  * Unified the `unit` parameter in the pyfar.dsp module to reduce duplicate code. Unit can now only be `samples` or `s` (seconds) but not `ms` or `mus` (milli, micro seconds) (PR #194)

* `pyfar.dsp.filter`

  * Add reconstructing fractional octave filter bank (PR #180)
  * Bugfix for mis-matching filter slopes in `crossover` filter (PR #174)

* Refactored internal handling of filter functionality for filter classes (PR #190)
* Added functionality to save/read filter objects to/from disk in `pyfar.io.read` and `pyfar.io.write` (PR #192, #182)
* Improved unit tests
* Improved documentation

0.1.0 (2021-04-11)
------------------
* First release on PyPI

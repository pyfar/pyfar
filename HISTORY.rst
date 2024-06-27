=======
History
=======

0.6.8 (2024-06-27)
------------------
* Make compatibile for scipy 1.14 (PR #638)
* Fix previous page title on pyfar main page (#637)
* Enhance user warning for writing clipped audio files (#624)
* Improve documentation (#634)
* Introduce radius_tol to Coordiantes.find_nearest (#621)

0.6.7 (2024-06-17)
------------------
* Make compatibile for numpy 2.0 (PR #629)

0.6.6 (2024-06-07)
------------------
* Improve documentation (PR #569, #574, #590, #591, #597, #605)
* Allow all arithmetic operations involving a single pyfar audio object (#606)
* Fix bug in `Coordinates.find_within` where not all or too many points were returned (#617)
* Allow `None` in `Coordinates.sh_order` property (#596)
* Always use `frequency_range` as parameter and deprecate `freq_range` in pyfar 0.8.0 (#589)
* Update dependencies (PR #564)
* Improve CI (#570, #586, #605, #607)

0.6.5 (2024-03-15)
------------------
* `Coordinates.show` now plots on equally scaled axis (PR #554)
* Update documentation to pydata theme (PR #560)
* Improve documentation (PR #544, #548, #549, #556)
* Add testing for Python 3.12 (PR #561)

0.6.4 (2024-02-16)
------------------
* Bugfix in `pyfar.io.read_comsol_header`: Fix for reading expressions containing the characters '-', '[', and ']' (PR #535)
* Bugfix in `pyfar.dsp.fft.normalization`: Correct the normalization factor for the case `fft_norm=='psd'`. (PR #541)
* Maintenance: Remove tests for deprecated numpy functionality (PR #537)

0.6.3 (2024-01-26)
------------------
* Bugfix in `pyfar.utils.concatenate_channels`: Amplitude of time domain Signals was wrong when concatenating in the frequency domain and concatenation failed for a mixture of Signals in the time and frequency domain (PR #532)

0.6.2 (2024-01-12)
------------------
* Bugfix in `pyfar.Coordinates.find_within`: Fix for spherical distance measures and improved flexibility and documentation (PR #524)

0.6.1 (2023-11-17)
------------------
* Bugfix in `pyfar.Coordinates.find_nearest`: Correct name of parameter 'spherical_radians' in docstring and fix computation of spherical distance between query points and actual points (PR #519)
* Improve `pyfar.signals.files.head_related_impulse_responses`: Use new structure of `pyfar.Coordinates` to find the requested head-related impulse responses (PR #520)

0.6.0 (2023-10-20)
------------------
* Refactored `pyfar.Coordinates` class and module

  * Added getter and setter for each pyfar coordinate, e.g., `pyfar.Coordinates.elevation` (PR #429)
  * Added getter and setter for each pyfar coordinate system, e.g., `pyfar.Coordinates.cartesian` (PR #429)
  * Added possibility to use an array of indices for `Coordinates.show` (PR #478)
  * Deprecated getter and setter methods `get_cart`, `set_cart`, `get_sph`, `set_sph`, `get_cyl`, `set_cyl`. Those will be removed in pyfar 0.8.0 (PR #429)
  * Deprecated the class property `Coordinates.sh_order`, which will be removed in pyfar 0.8.0 (PR #429) in favor of the `sampling_sphere` class from `spharpy v1.0.0 <https://spharpy.readthedocs.io/en/stable/>`_ (PR #429)
  * Added new class methods `Coordinates.find_nearest` and `Coordinates.find_within` (PR #429)
  * Deprecatex methods `Coordinates.find_nearest_k`, `Coordinates.find_slice`, `Coordinates.find_nearest_cart`, and `Coordinates.find_nearest_sph`. Will be removed in pyfar 0.8.0 (PR #478)
  * Added `rad2deg` and `deg2rad` converter (PR #500)
  * Coordinates angles are always returned in radians (PR #429)
  * Coordinates are always stored in cartesian coordinates internally and converted upon request (PR #429)
  * Changed type of return arguments in now deprecated `Coordinates.find_slice` (PR #386)

* pyfar audio classes (`pyfar.Signal`, `pyfar.TimeData`, `pyfar.FrequencyData`)

  * Added the possibility to store spectra with a single frequency (PR #433)
  * Empty comments, e.g., in `Signal.comment` are now set as an empty string not as 'none' (PR #379)
  * Deprecated the possibility to call `len(Signal)` because it was not clearly described and redundant (PR #418)

* `pyfar.utils`

  * Added functions to broadcast audio classes to a certain channel dimension or shape in `pf.utils.broadcast_cshape`, `pf.utils.broadcast_cshapes`, `pf.utils.broadcast_cdim`, `pf.utils.broadcast_cdims` (PR #385)

* `pyfar.dsp`

  * Added `pyfar.dsp.concatenate` function for pyfar audio objects (PR #452)
  * Added `pyfar.dsp.filter.notch` function (PR #441)
  * Added the possibility to cast signals with different channel dimensions in `pyfar.dsp.convolve` (PR #404)
  * Allowed NaN values in `pyfar.dsp.average` and `pyfar.dsp.normalize` (PR #425, #399)
  * Added more verbose names for the `mode` parameter of `pyfar.dsp.pad_zeros` (PR #381)

* `pyfar.plot`

  * Added the possibility to pass an empty dictionary as plot style to all pyfar plot function to use the currently active plot stlye in favor of the pyfar plot style (PR #446)
  * Removed unwanted minor ticks that could appear if zooming into a logarithmic frequency axis (PR #450)

* `pyfar.io`

  * Save the current pyfar version if using `pyfar.io.write` for providing more verbose feedback in case old data can not be read with newer versions of pyfar in the future (PR #445)
  * Updated version of sofar package. `pyfar.io.read_sofa` now also works with path objects (PR #472)
  * `pyfar.io.read_comsol` can now handle expressions containing the characters '*' '(' and ')' (PR #393)
  * `pyfar.io.write_audio` does now accept sampling rates of type float, if they do not contain decimal values (PR #414)

* `pyfar.signals.files`

  * Bugfix HRTFs are now returned in the requested order (PR #387)

* `pyfar.samplings`

  * Deprecated pyfar samplings in pyfar 0.8.0. Samplings and are now available from `spharpy v1.0.0 <https://spharpy.readthedocs.io/en/stable/>`_ (PR #486)

* Documentation

  * Show the plot shortcuts for interactive plotting (PR #422)
  * Added documentation for missing `unit` parameter in `pyfar.dsp.fractional_time_shift` (PR #484)
  * Corrected plot legend in the documentation of `pyfar.dsp.InterpolateSpectrum` (PR #457)
  * Improved documentation for `pyfar.dsp.filter.GammatoneBands` (PR #372)
  * Improved display of time axes in plots shown in the documentation (PR #423)
  * Add links to pyfar.org, readthedocs, and github on pypi.org (PR #356)
  * Improved documentation (PR #467, #458, #394, #498)

* CI, testing, and installation

  * Added `PyfarDeprecationWarning` Class to make sure warnings are always shown (PR #419, #397)
  * Made it possible to install and run pyfar in read only containers (PR #499)
  * Removed `tox.ini` which is not needed anymore after moving to circle CI (PR #480)
  * Updated testing guidelines (PR #407)
  * Adapted tests to avoid warnings from third party packages (PR #477, #434, #388)
  * Removed functions scheduled for deprecation in pyfar 0.6.0 (PR #476)
  * Added testing for Python 3.11 (PR #471)
  * Removed authors in favor of contributions shown on github (PR #413)


0.5.4 (2023-09-29)
------------------
* Dependencies: Constrain matplotlib to versions <= 3.7, due to deprecations of the tight_layout function in matplotlib 3.8 (PR #497).
* Bugfix: Fix order `order` property for `pyfar.FilterSOS` (PR #487).
* Bugfix: Fix broken tests for filter class copy methods (PR #488).
* Improvements to the documentation (PR #470).
* Flake8 fixes.

0.5.3 (2023-03-30)
------------------

* Bugfix: Spectrum interpolation on logarithmically spaced frequency bins including zero frequency. (PR #453)
* Bugfix: Include signal domain and fft norm when writing Signals to far-files. (PR #443)
* Bugfix: Return the HRIRs contained in the sample file in the correct order. (PR #448)

0.5.2 (2023-01-20)
------------------

* Bugfix: Remove deprecated usage of `np.int`. (PR #409)
* Bugfix: Switch to MathJax to fix equation rendering issues in the documentation. (PR #420)
* Bugfix: `read_comsol` now allow for `*`, `(`, and `)` in expressions and units. (PR #417, originally #393)
* Bugfix: `read_sofa` now support reading files of conventions `FIR-E` and `TF-E`. (PR  #415)
* General: Update information on PyPI. (PR #427, originally #377)

0.5.1 (2022-10-28)
------------------
* Bugfix: Allow setting of the sampling rate in GammatoneBands (PR #374)
* Bugfix: Added GammatoneBands filter bank to concepts (PR #372)


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
  * Added `average` function for averaging channels (PR #330)

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

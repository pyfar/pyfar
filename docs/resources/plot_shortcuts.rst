**Use these shortcuts to toggle between plots**

.. list-table::
   :widths: 25 100
   :header-rows: 1

   * - Key
     - Plot
   * - 1, shift+t
     - :py:func:`~pyfar.plot.time`
   * - 2, shift+f
     - :py:func:`~pyfar.plot.freq`
   * - 3, shift+p
     - :py:func:`~pyfar.plot.phase`
   * - 4, shift+g
     - :py:func:`~pyfar.plot.group_delay`
   * - 5, shift+s
     - :py:func:`~pyfar.plot.spectrogram`
   * - 6, ctrl+shift+t, ctrl+shift+f
     - :py:func:`~pyfar.plot.time_freq`
   * - 7, ctrl+shift+p
     - :py:func:`~pyfar.plot.freq_phase`
   * - 8, ctrl+shift+g
     - :py:func:`~pyfar.plot.freq_group_delay`

Note that not all plots are available for TimeData and FrequencyData objects as detailed in the :py:mod:`plot module <pyfar.plot>` documentation.

**Use these shortcuts to control the plot**

.. list-table::
   :widths: 25 100
   :header-rows: 1

   * - Key
     - Action
   * - left
     - move x-axis view to the left
   * - right
     - move x-axis view to the right
   * - up
     - move y-axis view upwards
   * - down
     - y-axis view downwards
   * - +, ctrl+shift+up
     - move colormap range up
   * - -, ctrl+shift+down
     - move colormap range down
   * - shift+right
     - zoom in x-axis
   * - shift+left
     - zoom out x-axis
   * - shift+up
     - zoom out y-axis
   * - shift+down
     - zoom in y-axis
   * - shift+plus, alt+shift+up
     - zoom colormap range in
   * - shift+minus, alt+shift+down
     - zoom colormap range out
   * - shift+x
     - toggle x-axis unit or scale (see below for more information)
   * - shift+y
     - toggle y-axis unit or scale (see below for more information)
   * - shift+c
     - toggle colormap unit (see below for more information)
   * - shift+m
     - toggle display of complex audio data (real, imaginary, or absolute time data; left or right-sided spectrum)
   * - shift+a
     - toggle between plotting all channels and plotting single channels
   * - <
     - cycle between line and 2D plots
   * - >
     - toggle between vertical and horizontal orientation of 2D plots
   * - ., ]
     - show next channel
   * - ,, [
     - show previous channel

**Notes on plot controls**

- Moving and zooming the x and y axes is supported by all plots.
- Moving and zooming the colormap is only supported by plots that have a colormap.
- Toggling the x-axis, y-axis and colormap toggles between

  - linear and logarithmic axis scaling for frequency axes,
  - seconds, milliseconds, microseconds, and samples for time axes,
  - linear amplitude and amplitude in dB for axes showing amplitudes,
  - wrapped and unwrapped phase for axes showing phase phase information.

- Toggling the x-axis style is supported by: :py:func:`~pyfar.plot.time`, :py:func:`~pyfar.plot.freq`, :py:func:`~pyfar.plot.phase`, :py:func:`~pyfar.plot.group_delay`, :py:func:`~pyfar.plot.spectrogram`, :py:func:`~pyfar.plot.time_freq`, :py:func:`~pyfar.plot.freq_phase`, :py:func:`~pyfar.plot.freq_group_delay` (and their 2d versions)
- Toggling the y-axis style is supported by: :py:func:`~pyfar.plot.time`, :py:func:`~pyfar.plot.freq`, :py:func:`~pyfar.plot.phase`, :py:func:`~pyfar.plot.group_delay`, :py:func:`~pyfar.plot.spectrogram`, :py:func:`~pyfar.plot.time_freq`, :py:func:`~pyfar.plot.freq_phase`, :py:func:`~pyfar.plot.freq_group_delay` (and their 2d versions)
- Toggling the colormap style is supported by all 2d plots
- Toggling between line and 2D plots is not supported by: :py:func:`~pyfar.plot.spectrogram`

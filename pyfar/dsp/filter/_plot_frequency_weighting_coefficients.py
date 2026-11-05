"""
Plot the precomputed frequency weighting coefficients from
``_frequency_weighting_coefficients.py`` for visual inspection.

Run this file as a module from the repository root to plot all sampling
rates, or call :py:func:`plot_frequency_weighting_coefficients` for more
control::

    python -m pyfar.dsp.filter._plot_frequency_weighting_coefficients

Running it by its file path instead puts ``pyfar/dsp/filter`` on the import
path rather than the repository root, so an installed (possibly older) pyfar
may be imported instead of this one.
"""
from typing import Literal, Optional
import numpy as np
import scipy.signal as sps
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pyfar as pf
from pyfar.dsp.filter._frequency_weighting_coefficients import (
    _FREQUENCY_WEIGHTING_SOS)

_COLORS = {"mean": "C0", "max": "C1"}


def plot_frequency_weighting_coefficients(
        mode: Literal["difference", "curve"] = "difference",
        sampling_rates: Optional[list[float]] = None,
        c_offset: float = 1,
        y_margin: float = 1,
        n_frequencies: int = 2**14,
        ):
    """
    Plot both candidates ("max" and "mean") of the precomputed A and C
    weighting filters, one figure per sampling rate.

    Parameters
    ----------
    mode : str, optional
        ``"difference"`` plots filter minus reference in dB, with the C
        weighting shifted by `c_offset` dB to separate it from A.
        ``"curve"`` plots the filter magnitude responses in dB together
        with the reference curves (black, semi-transparent).
        The default is ``"difference"``.
    sampling_rates : list of float, optional
        The sampling rates to plot. The default ``None`` plots all
        sampling rates for which coefficients are available.
    c_offset : float, optional
        Offset in dB added to the C weighting in ``"difference"`` mode.
        The default is ``1``.
    y_margin : float, optional
        Shown range in dB below 0 and above `c_offset` in ``"difference"``
        mode. The default is ``1``.
    n_frequencies : int, optional
        Number of logarithmically spaced frequencies between 1 Hz and the
        Nyquist frequency at which the filters are evaluated. The mean and
        max absolute errors in the legend are computed on the same grid.
        The default is ``2**14``.
    """
    if mode not in ["difference", "curve"]:
        raise ValueError("mode must be 'difference' or 'curve'")

    available = sorted({fs for _, fs in _FREQUENCY_WEIGHTING_SOS})
    if sampling_rates is None:
        sampling_rates = available

    for fs in sampling_rates:
        if fs not in available:
            raise ValueError(f"No coefficients for sampling rate {fs} Hz. "
                             f"Available: {available}")
        freqs = np.logspace(0, np.log10(fs / 2), n_frequencies)
        plt.figure(figsize=(6.7, 3.3))
        for weighting in ["A", "C"]:
            offset = c_offset if weighting == "C" and \
                mode == "difference" else 0
            linestyle = "-" if weighting == "A" else "--"
            reference = pf.constants.frequency_weighting_curve(
                weighting, freqs)
            if mode == "difference":
                plt.axhline(offset, color="k", linewidth=.5)
            else:
                plt.semilogx(freqs, reference, color="k", alpha=.5,
                             linestyle=linestyle,
                             label=f"{weighting}, reference")
            for criterion, sos in \
                    _FREQUENCY_WEIGHTING_SOS[(weighting, fs)].items():
                _, h = sps.sosfreqz(sos, freqs, fs=fs)
                levels = 20 * np.log10(np.abs(h))
                errors = np.abs(levels - reference)
                plotted = levels - reference + offset \
                    if mode == "difference" else levels
                plt.semilogx(
                    freqs, plotted, color=_COLORS[criterion],
                    linestyle=linestyle,
                    label=(f"{weighting}, best {criterion} "
                           f"(mean={np.mean(errors):.2e}, "
                           f"max={np.max(errors):.2e} dB)"))

        if mode == "difference":
            plt.title(f"fs = {fs:g} Hz (C shifted by +{c_offset} dB)")
            plt.ylabel("Filter - reference in dB")
            plt.ylim(-y_margin, c_offset + y_margin)
            plt.legend(fontsize=6, loc="upper left")
        else:
            plt.title(f"fs = {fs:g} Hz")
            plt.ylabel("Magnitude in dB")
            plt.xlim(20, fs / 2)
            plt.ylim(np.min(plotted) - 5, np.max(plotted) + 5)
            plt.legend(fontsize=6, loc="lower center")
        plt.xlabel("Frequency in Hz")
        plt.gca().xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x:g}"))
        plt.grid(which="both", alpha=.3)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    plot_frequency_weighting_coefficients()

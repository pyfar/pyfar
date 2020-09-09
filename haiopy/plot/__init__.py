# -*- coding: utf-8 -*-

"""Plot sub-package for haiopy."""

from .plot import (
    plot_freq,
    plot_time,
    plot_time_dB
)
from .spatial import (
    scatter,
    quiver
)

__all__ = [
    plot_freq,
    plot_time,
    plot_time_dB,
    scatter,
    quiver
]

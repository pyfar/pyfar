# -*- coding: utf-8 -*-
"""
This module contains functions for generating common deterministic and
stochastic audio signals such as impulses, sine sweeps, and noise signals.

All signal lengths are given in samples. The value for the length is casted to
an integer number in all cases. This makes it possible to pass float numbers.
For example:

``n_samples = .015 * sampling_rate``.
"""

from .deterministic import (
    sine, impulse, linear_sweep_time, exponential_sweep_time)

from .stochastic import (
    noise, pulsed_noise)

from . import files

__all__ = [
    'sine', 'impulse', 'noise', 'pulsed_noise',
    'linear_sweep_time', 'exponential_sweep_time',
    'files']

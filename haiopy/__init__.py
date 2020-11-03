# -*- coding: utf-8 -*-

"""Top-level package for haiopy."""

__author__ = """The haiopy developers"""
__email__ = ''
__version__ = '0.1.0'


from .haiopy import Signal
from .coordinates import Coordinates
from .orientations import Orientations

import haiopy.plot as plot
import haiopy.haiopy as haiopy


__all__ = ['Signal', 'Coordinates', 'Orientations', 'plot', 'haiopy']

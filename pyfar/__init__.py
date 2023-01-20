# -*- coding: utf-8 -*-

"""Top-level package for pyfar."""

__author__ = """The pyfar developers"""
__email__ = ''
__version__ = '0.5.2'


from .classes.audio import Signal, TimeData, FrequencyData
from .classes.audio import (add, subtract, multiply, divide, power,
                            matrix_multiplication)
from .classes.coordinates import Coordinates
from .classes.orientations import Orientations
from .classes.filter import FilterFIR, FilterIIR, FilterSOS

from . import plot
from . import samplings
from . import io
from . import dsp
from . import signals


__all__ = [
    'Signal',
    'TimeData',
    'FrequencyData',
    'add',
    'subtract',
    'multiply',
    'divide',
    'power',
    'matrix_multiplication',
    'Coordinates',
    'Orientations',
    'FilterFIR',
    'FilterIIR',
    'FilterSOS',
    'plot',
    'samplings',
    'io',
    'dsp',
    'signals']

# -*- coding: utf-8 -*-

"""Top-level package for pyfar."""

__author__ = """The pyfar developers"""
__email__ = ''
__version__ = '0.7.0'


from .classes.audio import Signal, TimeData, FrequencyData
from .classes.audio import (add, subtract, multiply, divide, power,
                            matrix_multiplication)
from .classes.coordinates import Coordinates
from .classes.coordinates import (deg2rad, rad2deg, dot, cross)
from .classes.orientations import Orientations
from .classes.filter import FilterFIR, FilterIIR, FilterSOS
from .classes.transmission_matrix import TransmissionMatrix

from . import plot
from . import samplings
from . import io
from . import dsp
from . import signals
from . import utils


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
    'deg2rad',
    'rad2deg',
    'Orientations',
    'FilterFIR',
    'FilterIIR',
    'FilterSOS',
    'TransmissionMatrix',
    'plot',
    'samplings',
    'io',
    'dsp',
    'signals',
    'utils',
    'dot',
    'cross',
    ]

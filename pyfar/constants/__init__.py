"""Module for constants and physical properties of air."""

from typing import Final

from .speed_of_sound import (
    speed_of_sound_simple,
)

__all__ = [
    'speed_of_sound_simple',
]


standard_atmosphere_pressure: Final[float] = 101325.0
"""
Standard atmosphere pressure in Pa as defined in [#]_.

.. math::
    atm = 101325 Pa

Returns
-------
float
    Standard atmosphere pressure in Pa.

References
----------
.. [#] https://en.wikipedia.org/wiki/Atmospheric_pressure

"""

absolute_zero_celsius: Final[float] = -273.15
"""
Absolute zero temperature in Celsius as defined in [#]_.

.. math::
    T_0 = -273.15 °C

Returns
-------
float
    Absolute zero temperature in Celsius.

References
----------
.. [#] https://en.wikipedia.org/wiki/Absolute_zero
"""

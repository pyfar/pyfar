"""Module for constants and physical properties of air."""

from typing import Final

from .speed_of_sound import (
    speed_of_sound_simple,
)

__all__ = [
    'speed_of_sound_simple',
]


# useful constants for acoustics, physics and engineering


standard_atmosphere_pressure: Final[float] = 101325.0
"""
Standard atmosphere pressure in Pa as defined in [#]_.

.. math::
    P_{atm} = 101325 Pa

Returns
-------
float
    Standard atmosphere pressure in Pa.

References
----------
.. [#] https://en.wikipedia.org/wiki/Atmospheric_pressure

"""


standard_air_density: Final[float] = 1.204
r"""
Standard air density in :math:`kg/m^3` at standard atmosphere pressure and 20°C
as defined in [#]_.

.. math::
    \rho_{atm} = 1.204 \frac{kg}{m^3}

Returns
-------
float
    Standard atmosphere temperature in Kelvin.

References
----------
.. [#] https://en.wikipedia.org/wiki/Density_of_air
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

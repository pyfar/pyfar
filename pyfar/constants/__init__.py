"""Module for constants and physical properties of air."""

from typing import Final

from .speed_of_sound import (
    speed_of_sound_simple,
)

__all__ = [
    'speed_of_sound_simple',
]


standard_atmosphere_pressure: Final[float] = 101325.0
r"""
Standard atmosphere pressure in Pa as defined in [#]_.

.. math::
    P_\text{atm} = 101325 \, \text{Pa}

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
Standard air density in :math:`\text{kg}/\text{m}^3` at standard
atmosphere pressure and 20°C as defined in [#]_.

.. math::
    \rho_\text{atm} = 1.204 \, \frac{\text{kg}}{\text{m}^3}

Returns
-------
float
    Standard air density in :math:`\text{kg}/\text{m}^3`.

References
----------
.. [#] https://en.wikipedia.org/wiki/Density_of_air
"""


absolute_zero_celsius: Final[float] = -273.15
r"""
Absolute zero temperature in Celsius as defined in [#]_.

.. math::
    T_0 = -273.15 \text{°C}

Returns
-------
float
    Absolute zero temperature in degree Celsius.

References
----------
.. [#] https://en.wikipedia.org/wiki/Absolute_zero
"""

reference_sound_pressure: Final[float] = 20e-6
r"""
Reference sound pressure :math:`p_\text{ref}` in Pa as defined in [#]_.

.. math::
    p_\text{ref} = 20 \, \mathrm{\mu Pa}

Returns
-------
float
    Reference sound pressure in Pascal.

References
----------
.. [#] https://en.wikipedia.org/wiki/Sound_pressure

"""

reference_sound_power: Final[float] = 1e-12
r"""
Reference sound power :math:`P_\text{ref}` in W as defined in [#]_.

.. math::
    P_\text{ref} = 1 \, \text{pW} = 10^{-12} \, \text{W}

Returns
-------
float
    Reference sound power in Watt.

References
----------
.. [#] https://en.wikipedia.org/wiki/Sound_power

"""

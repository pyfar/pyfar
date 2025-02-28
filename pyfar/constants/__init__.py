"""Module for constants and physical properties of air."""

from typing import Final

from .speed_of_sound import (
    speed_of_sound_simple,
)

__all__ = [
    'speed_of_sound_simple',
]


standard_atmospheric_pressure: Final[float] = 101325.0
r"""
Standard atmospheric pressure in Pa as defined in [#]_.

.. math::
    P_\text{atm} = 101325 \, \text{Pa}

Returns
-------
float
    Standard atmospheric pressure in Pa.

References
----------
.. [#] https://en.wikipedia.org/wiki/Atmospheric_pressure

"""


absolute_zero_celsius: Final[float] = -273.15
r"""
Absolute zero temperature in Celsius [#]_.

.. math::
    t_0 = -273.15 \text{°C}

Returns
-------
t_0 : float
    Absolute zero temperature in degree Celsius.

References
----------
.. [#]  https://en.wikipedia.org/wiki/Absolute_zero
"""


reference_sound_pressure: Final[float] = 20e-6
r"""
Reference sound pressure :math:`p_\text{ref}` in Pa [#]_.

.. math::
    p_\text{ref} = 20 \, \mathrm{\mu Pa}

Returns
-------
p_ref : float
    Reference sound pressure in Pascal.

References
----------
.. [#] https://asastandards.org/terms/reference-value-for-sound-pressure-2/

"""

reference_sound_power: Final[float] = 1e-12
r"""
Reference sound power :math:`P_\text{ref}` in W [#]_.

.. math::
    P_\text{ref} = 1 \, \text{pW} = 10^{-12} \, \text{W}

Returns
-------
P_ref : float
    Reference sound power in Watt.

References
----------
.. [#] https://asastandards.org/terms/sound-power-level/

"""

reference_air_temperature_celsius: Final[float] = 20
r"""
Reference air temperature in degree Celsius [#]_.

.. math::
    t_\text{ref} = 20 \text{°C}

Returns
-------
t_ref : float
    Reference air temperature in degree Celsius.

References
----------
.. [#] ISO 9613-1:1993, Acoustics -- Attenuation of sound during
       propagation outdoors -- Part 1: Calculation of the absorption of
       sound by the atmosphere.
"""

reference_speed_of_sound: Final[float] = 343.2
r"""
Reference speed of sound :math:`c_\text{ref}` in m/s for 20°C and dry air [#]_.

.. math::
    c_\text{ref} = 343.2 \, \text{m/s}

Returns
-------
c_ref : float
    Reference speed of sound in m/s.

References
----------
.. [#] ISO 9613-1:1993, Acoustics -- Attenuation of sound during
       propagation outdoors -- Part 1: Calculation of the absorption of
       sound by the atmosphere.
"""

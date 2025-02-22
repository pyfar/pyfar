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
    t_0 = -273.15 \text{°C}

Returns
-------
t_0 : float
    Absolute zero temperature in degree Celsius.

References
----------
.. [#]  ISO 9613-1:1993, Acoustics -- Attenuation of sound during
        propagation outdoors -- Part 1: Calculation of the absorption of
        sound by the atmosphere.
"""


reference_sound_pressure: Final[float] = 20e-6
r"""
Reference sound pressure :math:`p_\text{ref}` in Pa as defined in [#]_.

.. math::
    p_\text{ref} = 20 \, \mathrm{\mu Pa}

Returns
-------
p_ref : float
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
P_ref : float
    Reference sound power in Watt.

References
----------
.. [#] https://en.wikipedia.org/wiki/Sound_power

"""

reference_air_temperature_celsius: Final[float] = 20
r"""
Reference air temperature in degree Celsius as defined in [#]_.

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
Reference speed of sound :math:`c_\text{ref}` in m/s for 20°C and dry air
as defined in [#]_.

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

reference_air_impedance: Final[float] = (
    reference_speed_of_sound * standard_air_density)
r"""
Reference air impedance :math:`Z_\text{ref}` in Pa s/m as defined in [#]_.

.. math::
    Z_\text{ref} = \rho_\text{ref} \cdot c_\text{ref} \approx 413.2 \text{Pa s/m}

Returns
-------
Z_ref : float
    Reference air impedance in Pa s/m.

References
----------
.. [#] https://en.wikipedia.org/wiki/Acoustic_impedance
"""

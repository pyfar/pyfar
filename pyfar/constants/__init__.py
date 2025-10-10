# Module for constants and physical properties of air.  # noqa: D104

from typing import Final

from .constants import (
    saturation_vapor_pressure_magnus,
    density_of_air,
    fractional_octave_filter_tolerance,
)

from .speed_of_sound import (
    speed_of_sound_simple,
    speed_of_sound_cramer,
    speed_of_sound_ideal_gas,
)

from .constants import (
    air_attenuation,
)

from .frequency_weighting import (
    frequency_weighting_curve,
    frequency_weighting_band_corrections,
)

__all__ = [
    'saturation_vapor_pressure_magnus',
    'density_of_air',
    'speed_of_sound_simple',
    'speed_of_sound_cramer',
    'speed_of_sound_ideal_gas',
    'air_attenuation',
    'frequency_weighting_curve',
    'frequency_weighting_band_corrections',
    'fractional_octave_filter_tolerance',
]


reference_atmospheric_pressure: Final[float] = 101325.0
r"""
Reference atmospheric pressure :math:`P_\text{atm}` in Pascal [#]_.

.. math::
    P_\text{atm} = 101325 \, \text{Pa}

Returns
-------
float
    Reference atmospheric pressure in Pascal.

References
----------
.. [#] https://en.wikipedia.org/wiki/Atmospheric_pressure

"""


absolute_zero_celsius: Final[float] = -273.15
r"""
Absolute zero temperature :math:`\mathrm{t_0}` in degree Celsius [#]_.

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
Reference sound pressure :math:`p_\text{ref}` in Pascal [#]_.

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
Reference sound power :math:`P_\text{ref}` in Watt [#]_.

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
Reference air temperature :math:`t_\text{ref}` in degree Celsius [#]_.

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


reference_air_density: Final[float] = 1.204
r"""
Reference air density in :math:`\text{kg}/\text{m}^3` at reference
atmospheric pressure and 20°C [#]_.

.. math::

    \rho_\text{atm} = 1.204 \, \frac{\text{kg}}{\text{m}^3}


Returns
-------
float
    Reference air density in :math:`\text{kg}/\text{m}^3`.

References
----------
.. [#] https://en.wikipedia.org/wiki/Density_of_air

"""


reference_air_impedance: Final[float] = (
    reference_speed_of_sound * reference_air_density)
r"""
Reference air impedance :math:`Z_\text{ref}` in Pa s/m is calculated based on
:py:attr:`reference_speed_of_sound` and
:py:attr:`standard_air_density`.

.. math::

    Z_\text{ref} = \rho_\text{atm} \cdot c_\text{ref} \approx 413.2
    \, \text{Pa s/m}

Returns
-------
Z_ref : float
    Reference air impedance in Pa s/m.

"""

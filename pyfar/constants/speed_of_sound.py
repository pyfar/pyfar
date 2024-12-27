"""File containing all speed of sound calculation functions."""
import numpy as np
from . import utils


def speed_of_sound_simple(temperature):
    r"""
    Calculate the speed of sound in air using a simplified version
    based on the temperature.

    Calculation is in accordance with ISO 9613-1 [#]_, as described in
    ISO 17497-1 [#]_.

    .. math::

        c = 343.2 \cdot \sqrt{\frac{t + 273.15}{293.15}} m/s

    Parameters
    ----------
    temperature : float, array_like
        Temperature in degree Celsius from -20°C to +50°C.

    Returns
    -------
    speed_of_sound : float, array_like
        Speed of sound in air in (m/s).

    References
    ----------
    .. [#] ISO 9613-1:1993, Acoustics -- Attenuation of sound during
           propagation outdoors -- Part 1: Calculation of the absorption of
           sound by the atmosphere.
    .. [#] ISO 17497-1:2004, Acoustics -- Sound-scattering properties of
           surfaces -- Part 1: Measurement of the random-incidence scattering
           coefficient in a reverberation room.
    """
    if np.any(np.array(temperature) < -20) or np.any(
            np.array(temperature) > 50):
        raise ValueError("Temperature must be between -20°C and +50°C.")
    temperature = np.array(temperature, dtype=float) if isinstance(
        temperature, list) else temperature
    return 343.2*np.sqrt((temperature+273.15)/293.15)


def speed_of_sound_ideal_gas(
        temperature, relative_humidity, atmospheric_pressure):
    """Calculate speed of sound in air using the ideal gas law.

    The speed of sound in air can be calculated based on chapter 6.3 in [#]_.

    Parameters
    ----------
    temperature : float, array_like
        Temperature in degree Celsius.
    relative_humidity : float, array_like
        Relative humidity in the range of 0 to 1.
    atmospheric_pressure : float, optional
        Atmospheric pressure in pascal, by default 101325 Pa.
        It must be between ?? to ?? Pa.

    Returns
    -------
    speed_of_sound : float, array_like
        Speed of sound in air in (m/s).

    References
    ----------
    .. [#] V. E. Ostashev and D. K. Wilson, Acoustics in Moving Inhomogeneous
           Media, 2nd ed. London: CRC Press, 2015. doi: 10.1201/b18922.

    """

    P = atmospheric_pressure

    # Constants according to 6.3.2
    R = 8.314  # J/(K mol)
    gamma_a = 1.400
    gamma_w = 1.330
    mu_a = 28.97  # g/mol
    mu_w = 18.02  # g/mol
    R_a = R/mu_a

    # partial pressure of water vapor in Pa
    p = utils.saturation_vapor_pressure(temperature)
    e = relative_humidity * p

    alpha = mu_a / mu_w
    # Equation 6.80
    delta = (1 - (1/gamma_a)) / (1 - (1/gamma_w))
    nu = (gamma_a - 1) / (gamma_w - 1)

    # Equation 6.70
    C = (e/P) / (alpha * (1 - e/P))

    return np.sqrt(
        gamma_a * R_a * temperature * (1 + (alpha * (1 + delta - nu) - 1) * C),
    )

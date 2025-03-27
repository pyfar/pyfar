"""File containing all speed of sound calculation functions."""
import numpy as np
from . import constants
import pyfar as pf


def speed_of_sound_simple(temperature):
    r"""
    Calculate the speed of sound in air using a simplified version
    of the ideal gas law based on the temperature.

    The calculation follows ISO 9613-1 [#]_ (Formula A.5).

    .. math::

        c(t) = c_\text{ref} \cdot \sqrt{\frac{t - t_0}{t_\text{ref} - t_0}}

    where:
        - :math:`t` is the air temperature (°C)
        - :math:`t_\text{ref}=20\mathrm{°C}` is the reference air temperature
          (°C), see
          :py:attr:`reference_air_temperature_celsius`
        - :math:`t_0=-273.15` °C is the absolute zero temperature (°C), see
          :py:attr:`absolute_zero_celsius`
        - :math:`c=343.2` m/s is the speed of sound at the reference
          temperature, see :py:attr:`reference_speed_of_sound`

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
    """
    # input validation
    if np.any(np.array(temperature) < -20) or np.any(
            np.array(temperature) > 50):
        raise ValueError("Temperature must be between -20°C and +50°C.")
    # convert to numpy array if necessary
    temperature = np.array(temperature, dtype=float) if isinstance(
        temperature, list) else temperature

    t_ref = pf.constants.reference_air_temperature_celsius
    t_0 = pf.constants.absolute_zero_celsius
    c_ref = pf.constants.reference_speed_of_sound
    return c_ref*np.sqrt((temperature-t_0)/(t_ref-t_0))


def speed_of_sound_ideal_gas(
        temperature, relative_humidity, atmospheric_pressure=None,
        saturation_vapor_pressure=None):
    """Calculate speed of sound in air using the ideal gas law.

    The speed of sound in air can be calculated based on chapter 6.3 in [#]_.
    All input parameters must be broadcastable to the same shape.

    Parameters
    ----------
    temperature : float, array_like
        Temperature in degree Celsius.
    relative_humidity : float, array_like
        Relative humidity in the range of 0 to 1.
    atmospheric_pressure : float, array_like, optional
        Atmospheric pressure in pascal, by default
        :py:attr:`reference_atmospheric_pressure`
    saturation_vapor_pressure : float, array_like, optional
        Saturation vapor pressure in Pa.
        If not given, the function
        :py:func:`~pyfar.constants.saturation_vapor_pressure` is used.
        Note that the valid temperature range is therefore also dependent on
        :py:func:`~pyfar.constants.saturation_vapor_pressure`.

    Returns
    -------
    speed_of_sound : float, array_like
        Speed of sound in air in (m/s).

    References
    ----------
    .. [#] V. E. Ostashev and D. K. Wilson, Acoustics in Moving Inhomogeneous
           Media, 2nd ed. London: CRC Press, 2015. doi: 10.1201/b18922.
    """
    if atmospheric_pressure is None:
        atmospheric_pressure = pf.constants.reference_atmospheric_pressure
    # check inputs
    if not isinstance(temperature, (int, float, np.ndarray, list, tuple)):
        raise TypeError(
            'Temperature must be a number or array of numbers')
    if not isinstance(
            relative_humidity, (int, float, np.ndarray, list, tuple)):
        raise TypeError(
            'Relative humidity must be a number or array of numbers')
    if not isinstance(
            atmospheric_pressure, (int, float, np.ndarray, list, tuple)):
        raise TypeError(
            'Atmospheric pressure must be a number or array of numbers')
    temperature = np.array(temperature, dtype=float)
    temperature_kelvin = temperature - pf.constants.absolute_zero_celsius
    if np.any(np.array(temperature_kelvin) < 0):
        raise ValueError("Temperature must be above -273.15°C.")
    if np.any(np.array(relative_humidity) < 0) or np.any(
            np.array(relative_humidity) > 1):
        raise ValueError("Relative humidity must be between 0 and 1.")
    if np.any(np.array(atmospheric_pressure) < 0):
        raise ValueError("Atmospheric pressure must be larger than 0 Pa.")

    P = np.array(atmospheric_pressure, dtype=float)  # Pa
    relative_humidity = np.array(relative_humidity, dtype=float)

    # Constants according to 6.3.2
    R = 8.314  # J/(K mol)
    gamma_a = 1.400
    gamma_w = 1.330
    mu_a = 28.97*1e-3  # kg/mol
    mu_w = 18.02*1e-3  # kg/mol
    R_a = R / mu_a

    # partial pressure of water vapor in Pa
    if saturation_vapor_pressure is None:
        p = constants.saturation_vapor_pressure_magnus(temperature)  # Pa
    else:
        p = np.array(saturation_vapor_pressure, dtype=float)  # Pa
    e = relative_humidity * p  # Pa

    # next to Equation 6.69
    alpha = mu_a / mu_w
    # Equation 6.80
    delta = (1 - (1/gamma_a)) / (1 - (1/gamma_w))
    nu = (gamma_a - 1) / (gamma_w - 1)

    # Equation 6.70
    C = (e/P) / (alpha * (1 - e/P))

    # Equation 6.84
    return np.sqrt(
        gamma_a * R_a * temperature_kelvin * (
            1 + (alpha * (1 + delta - nu) - 1) * C),
    )

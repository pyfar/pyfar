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
        Atmospheric pressure in Pascal, by default
        :py:attr:`reference_atmospheric_pressure`.
    saturation_vapor_pressure : float, array_like, optional
        Saturation vapor pressure in Pascal.
        If not given, the function
        :py:func:`~pyfar.constants.saturation_vapor_pressure_magnus` is used.
        Note that the valid temperature range therefore also depends on
        :py:func:`~pyfar.constants.saturation_vapor_pressure_magnus`.

    Returns
    -------
    speed_of_sound : float, array_like
        Speed of sound in air in m/s.

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
    if np.any(np.array(atmospheric_pressure) <= 0):
        raise ValueError("Atmospheric pressure must be greater than 0 Pa.")
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


def speed_of_sound_cramer(
        temperature, relative_humidity, co2_ppm=425.19,
        atmospheric_pressure=None,
        ):
    r"""
    Calculate the speed of sound in air based on temperature, atmospheric
    pressure, humidity and CO\ :sub:`2` concentration.

    This implements Cramers method described in [#]_.

    Parameters
    ----------
    temperature : float, array_like
        Temperature in degree Celsius from 0°C to 30°C.
    relative_humidity : float, array_like
        Relative humidity in the range of 0 to 1.
    co2_ppm : float, array_like, optional
        CO\ :sub:`2` concentration in parts per million. The default is
        425.19 ppm, based on [#]_. Value must be below
        10 000 ppm (1%).
    atmospheric_pressure : float, array_like, optional
        Atmospheric pressure in Pascal, by default
        :py:attr:`reference_atmospheric_pressure`.
        Value must be between 75 000 Pa to 102000 Pa.

    Returns
    -------
    speed_of_sound : float, array_like
        Speed of sound in air in m/s.

    References
    ----------
    .. [#] O. Cramer, “The variation of the specific heat ratio and the
           speed of sound in air with temperature, pressure, humidity, and
           CO\ :sub:`2` concentration,” The Journal of the Acoustical Society
           of America, vol. 93, no. 5, pp. 2510-2516, May 1993,
           doi: 10.1121/1.405827.
    .. [#] Lan, X., Tans, P. and K.W. Thoning: Trends in globally-averaged
           CO2 determined from NOAA Global Monitoring Laboratory measurements.
           Version Friday, 14-Mar-2025 11:33:44 MDT
           https://doi.org/10.15138/9N0H-ZH07

    """
    if atmospheric_pressure is None:
        atmospheric_pressure = pf.constants.reference_atmospheric_pressure
    # convert to array
    temperature = np.array(temperature)
    relative_humidity = np.array(relative_humidity)
    atmospheric_pressure = np.array(atmospheric_pressure)
    co2_ppm = np.array(co2_ppm)

    # check inputs:
    if np.any(temperature < 0) or np.any(temperature > 30):
        raise ValueError("Temperature must be between 0°C and 30°C.")
    if np.any(relative_humidity < 0) or np.any(relative_humidity > 1):
        raise ValueError("Relative humidity must be between 0 and 1.")
    if np.any(atmospheric_pressure < 75e3) or np.any(
            atmospheric_pressure > 102e3):
        raise ValueError(
            "Atmospheric pressure must be between 75 000 Pa to 102 000 Pa.")
    if np.any(co2_ppm < 0) or np.any(co2_ppm > .01e6):
        raise ValueError(
            "CO2 concentration (ppm) must be between 0 ppm to 10 000 ppm.")

    # mole fraction of CO2 in air
    x_c = co2_ppm * 1e-6
    # pressure(atmospheric pressure) in Pa
    p = atmospheric_pressure
    # relative humidity as a fraction
    h = relative_humidity
    # temperature in Celsius
    t = temperature
    # thermodynamic temperature
    T = t - pf.constants.absolute_zero_celsius

    # enhancement factor f (Equation A2)
    f = 1.00062+3.14e-8 * p + 5.6e-7*temperature**2
    # saturation vapor pressure in air (Equation A3)
    p_sv = np.exp(1.2811805e-5*T**2-1.9509874e-2*T+34.04926034-6.3536311e3/T)
    # water vapor mole fraction (Equation A1)
    x_w = h * f * p_sv / p

    # check water mole fraction after Cramer
    if np.any(x_w < 0) or np.any(x_w > 0.06):
        raise ValueError("Water mole fraction must be between 0 and 0.06.")

    # Coefficients according to Cramer (Table. III)
    a0 = 331.5024
    a1 = 0.603055
    a2 = -0.000528
    a3 = 51.471935
    a4 = 0.1495874
    a5 = -0.000782
    a6 = -1.82e-7
    a7 = 3.73e-8
    a8 = -2.93e-10
    a9 = -85.20931
    a10 = -0.228525
    a11 = 5.91e-5
    a12 = -2.835149
    a13 = -2.15e-13
    a14 = 29.179762
    a15 = 0.000486

    # approximation for c according to Cramer (Eq. 15)
    c1 = a0+a1*t+a2*t**2
    c2 = (a3+a4*t+a5*t**2)*x_w
    c3 = (a6+a7*t+a8*t**2)*p
    c4 = (a9+a10*t+a11*t**2)*x_c
    c5 = a12*x_w**2 + a13*p**2 + a14*x_c**2 + a15*x_w*p*x_c

    return c1 + c2 + c3 + c4 + c5

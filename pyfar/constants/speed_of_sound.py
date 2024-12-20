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


def speed_of_sound_cramer(
        temperature, relative_humidity, atmospheric_pressure=101325,
        c02_ppm=315):
    """
    Get speed of sound using Cramers method described in [#]_.

    The conditions are as follows:
    - for temperatures from 0°C Cto 30°C
    - atmospheric pressures from 75 000 Pa to 102000 Pa
    - up to 0.06 H20 mole fraction
    - CO2 concentrations up to 10 000 ppm
    https://www.kane.co.uk/knowledge-centre/what-are-safe-levels-of-co-and-co2-in-rooms

    Parameters
    ----------
    temperature : float, array_like
        Temperature in degree Celsius from 0°C Cto 30°C.
    relative_humidity : float, array_like
        Relative humidity in the range of 0 to 1.
    atmospheric_pressure : int, optional
        Atmospheric pressure in pascal, by default 101325 Pa.
        It must be between 75 000 Pa to 102000 Pa.
    c02_ppm : float, array_like
        co2 concentration in parts per million.


    References
    ----------
    .. [#] O. Cramer, “The variation of the specific heat ratio and the
           speed of sound in air with temperature, pressure, humidity, and CO2
           concentration,” The Journal of the Acoustical Society of America,
           vol. 93, no. 5, pp. 2510-2516, May 1993, doi: 10.1121/1.405827.

    """
    rel_hum = relative_humidity
    p_stat = atmospheric_pressure

    # the carbon dioxide mole fraction
    x_c = c02_ppm / 1e6

    # calculate saturation_vapor_pressure from Magnus Formula
    pws = utils.saturation_vapor_pressure(temperature) * 100

    # water vapor mole fraction
    xw = rel_hum * pws / p_stat

    # % Coefficients according to Cramer (Table. III)
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

    # % approximation for c according to Cramer (Eq. 15)
    c1 = a0+a1*temperature+a2*temperature**2
    c2 = (a3+a4*temperature+a5*temperature**2)*xw
    c3 = (a6+a7*temperature+a8*temperature**2)*p_stat
    c4 = (a9+a10*temperature+a11*temperature**2)*x_c
    c5 = a12*xw**2 + a13*p_stat**2 + a14*x_c**2 + a15*xw*p_stat*x_c

    return c1 + c2 + c3 + c4 + c5

"""Constant calculation."""
import numpy as np


def saturation_vapor_pressure_magnus(temperature):
    r"""
    Calculate the saturation vapor pressure of water in Pa using the
    Magnus formula.

    The Magnus formula is valid for temperatures between -45°C and 60°C [#]_.

    .. math::

        e_s = 610.94 \cdot \exp\left(\frac{17.625 \cdot T}{T + 243.04}\right)


    Parameters
    ----------
    temperature : float, array_like
        Temperature in degrees Celsius (°C).

    Returns
    -------
    p_sat : float, array_like
        Saturation vapor pressure in Pa.

    References
    ----------
    .. [#] O. A. Alduchov and R. E. Eskridge, “Improved Magnus Form
           Approximation of Saturation Vapor Pressure,” Journal of Applied
           Meteorology and Climatology, vol. 35, no. 4, pp. 60-609, Apr. 1996
    """

    if not isinstance(temperature, (int, float, np.ndarray, list, tuple)):
        raise TypeError(
            'temperature must be a number or array of numbers')
    if np.any(np.array(
            temperature) < -45) or np.any(np.array(temperature) > 60):
        raise ValueError("Temperature must be in the range of -45°C and 60°C.")
    if isinstance(temperature, (np.ndarray, list, tuple)):
        temperature = np.asarray(temperature, dtype=float)

    return 100 * 6.1094 * np.exp(
        (17.625 * temperature) / (temperature + 243.04))


def density_of_air(
        temperature, relative_humidity, atmospheric_pressure=None):
    r"""
    Calculate the density of air in kg/m³ based on the temperature,
    relative humidity, and atmospheric pressure.

    The density of air can be calculated based on chapter 6.3 in [#]_.
    All input parameters must be broadcastable to the same shape.

    Parameters
    ----------
    temperature : float, array_like
        Temperature in degrees Celsius (°C).
    relative_humidity : float, array_like
        Relative humidity in the range of 0 to 1.
    atmospheric_pressure : float, array_like, optional
        Atmospheric pressure in pascal (Pa), by default
        :py:attr:`reference_atmospheric_pressure`.

    Returns
    -------
    density : float, array_like
        Density of air in kg/m³.

    References
    ----------
    .. [#] V. E. Ostashev and D. K. Wilson, Acoustics in Moving Inhomogeneous
           Media, 2nd ed. London: CRC Press, 2015. doi: 10.1201/b18922.
    """
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
    if np.any(np.array(relative_humidity) < 0) or np.any(
            np.array(relative_humidity) > 1):
        raise ValueError("Relative humidity must be between 0 and 1.")
    if np.any(np.array(atmospheric_pressure) < 0):
        raise ValueError("Atmospheric pressure must be larger than 0 Pa.")

    P = np.array(atmospheric_pressure, dtype=float)  # Pa
    relative_humidity = np.array(relative_humidity, dtype=float)

    # Constants according to 6.3.2
    R = 8.314  # J/(K mol)
    mu_a = 28.97*1e-3  # kg/mol
    mu_w = 18.02*1e-3  # kg/mol
    R_a = R / mu_a

    # next to Equation 6.69
    alpha = mu_a / mu_w

    # partial pressure of water vapor in Pa
    p = saturation_vapor_pressure_magnus(temperature)  # Pa
    e = relative_humidity * p  # Pa

    # Equation 6.70
    C = (e/P) / (alpha * (1 - e/P))

    # Equation 6.71
    return P / (R_a * (temperature + 273.15)) * (1+C)/(1+alpha*C)

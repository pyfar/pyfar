"""File containing all speed of sound calculation functions."""
import numpy as np


def speed_of_sound_simple(temperature):
    r"""
    Calculate the speed of sound in air using a simplified version
    of the ideal gas law based on the temperature.

    The calculation follows ISO 9613-1 [#]_ (Formula A.5).

    .. math::

        c = 343.2 \cdot \sqrt{\frac{t + 273.15}{t_0 + 273.15}} \mathrm{m/s}

    where:
        - :math:`t` is the air temperature (°C)
        - :math:`t_0=20\mathrm{°C}` is the reference air temperature (°C)

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

    t_0 = 20
    return 343.2*np.sqrt((temperature+273.15)/(t_0+273.15))

"""File containing all speed of sound calculation functions."""
import numpy as np


def speed_of_sound_simple(temperature):
    r"""
    Calculate the speed of sound in air using a simplified version
    of the ideal gas law based on the temperature.

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

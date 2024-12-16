"""Air attenuation calculation."""
import numpy as np


def air_attenuation_iso(
        temperature, frequency, relative_humidity,
        atmospheric_pressure=101325, calculate_accuracy=False):
    """Calculate the pure tone air attenuation of sound in air according to
    ISO 9613-1.

    Calculation is in accordance with ISO 9613-1 [#]_. The shapes of
    ``temperature``, ``frequency``, ``relative_humidity``, and
    ``atmospheric_pressure`` must be broadcastable.

    Parameters
    ----------
    temperature : float, array_like
        Temperature in degree Celsius.
        Must be in the range of -20°C to 50°C for accuracy of +/-10% or
        must be greater than -70°C for accuracy of +/-50%.
    frequency : float, array_like
        Frequency in Hz. Must be greater than 50 Hz.
    relative_humidity : float, array_like
        Relative humidity in the range of 0.1 to 1.
    atmospheric_pressure : int, optional
        Atmospheric pressure in pascal, by default 101325 Pa.
    calculate_accuracy : bool, optional
        Weather to compute the accuracy or not. Default is not.

    Returns
    -------
    air_attenuation : float, array_like
        Pure tone air attenuation coefficient in decibels per meter for
        atmospheric absorption.
    accuracy : float, array_like, optional
        accuracy of the results according to the standard, if
        ``calculate_accuracy`` is True:

        ``10``, +/- 10% accuracy
            - molar concentration of water vapour: 0,05% to 5 %.
            - air temperature: 253,15 K to 323,15 (-20 °C to +50°C)
            - atmospheric pressure: less than 200 Pa (2 atm)
            - frequency-to-pressure ratio: 4 x 10-4 Hz/Pa to 10 Hz/Pa.

        ``20``, +/- 20% accuracy
            - molar concentration of water vapour: 0,005 % to 0,05 %,
              and greater than 5%
            - air temperature: 253,15 K to 323,15 (-20 °C to +50°C)
            - air temperature: greater than 200 K (- 73 °C)
            - frequency-to-pressure ratio: 4 x 10-4 Hz/Pa to 10 Hz/Pa.

        ``50``, +/- 50% accuracy
            - molar concentration of water vapour: less than 0,005%
            - air temperature: greater than 200 K (- 73 °C)
            - atmospheric pressure: less than 200 kPa (2 atm)
            - frequency-to-pressure ratio: 4 x 10-4 Hz/Pa to 10 Hz/Pa.

        ``-1``, no valid result
            else.

    References
    ----------
    .. [#] ISO 9613-1:1993, Acoustics -- Attenuation of sound during
           propagation outdoors -- Part 1: Calculation of the absorption of
           sound by the atmosphere.
    """
    # check inputs
    if not isinstance(temperature, (int, float, np.ndarray, list, tuple)):
        raise TypeError(
            'temperature must be a number or array of numbers')
    if not isinstance(frequency, (int, float, np.ndarray, list, tuple)):
        raise TypeError(
            'frequency must be a number or array of numbers')
    if not isinstance(
            relative_humidity, (int, float, np.ndarray, list, tuple)):
        raise TypeError(
            'relative_humidity must be a number or array of numbers')
    if not isinstance(
            atmospheric_pressure, (int, float, np.ndarray, list, tuple)):
        raise TypeError(
            'atmospheric_pressure must be a number or array of numbers')
    if not isinstance(calculate_accuracy, bool):
        raise TypeError(
            'calculate_accuracy must be a bool.')

    # check limits
    if np.any(np.array(temperature) < -73):
        raise ValueError("Temperature must be between -73°C.")
    if np.any(np.array(frequency) < 50):
        raise ValueError("Frequency must be greater than 50 Hz.")
    if np.any(np.array(relative_humidity) < .1) or np.any(
            np.array(relative_humidity) > 1):
        raise ValueError("Relative humidity must be between 0.1 and 1.")
    if np.any(np.array(atmospheric_pressure) > 200000):
        raise ValueError("Atmospheric pressure must less than 200 kPa.")

    # convert arrays
    all_are_singles = isinstance(temperature, (float, int))
    all_are_singles = all_are_singles and isinstance(frequency, (float, int))
    all_are_singles = all_are_singles and isinstance(
        relative_humidity, (float, int))
    if not all_are_singles:
        shape = np.broadcast_shapes(
            np.array(temperature).shape,
            np.array(frequency).shape,
            np.array(relative_humidity).shape,
            np.array(atmospheric_pressure).shape)
        temperature = np.broadcast_to(
            np.array(temperature, dtype=float), shape)
        frequency = np.broadcast_to(
            np.array(frequency, dtype=float), shape)
        relative_humidity = np.broadcast_to(
            np.array(relative_humidity, dtype=float), shape)
        atmospheric_pressure = np.broadcast_to(
            np.array(atmospheric_pressure, dtype=float), shape)

    # calculate air attenuation
    p_atmospheric_ref = 101325
    t_degree_ref = 20

    p_a = atmospheric_pressure
    p_r = p_atmospheric_ref
    f = frequency
    T = temperature + 273.15
    T_0 = t_degree_ref + 273.15
    p_sat_water = _p_sat_water(temperature)

    p_vapor = relative_humidity*p_sat_water
    # molar concentration of water vapor as a percentage
    h = p_vapor/p_a*10000

    # Oxygen relaxation frequency (Eq. 3)
    f_rO = (p_a/p_r)*(24+4.04e4*h*(0.02+h)/(0.391+h))
    # Nitrogen relaxation frequency (Eq. 4)
    f_rN = (p_a/p_r)*(T/T_0)**(-1/2)*(9+280*h*np.exp(
        -4.17*((T/T_0)**(-1/3)-1)))

    air_attenuation = 8.686*f**2*((1.84e-11*p_r/p_a*(T/T_0)**(1/2)) + \
        (T/T_0)**(-5/2)*(0.01275*np.exp(-2239.1/T)*(f_rO + (f**2/f_rO))**(-1)
        +0.1068*np.exp(-3352/T) * (f_rN + (f**2/f_rN))**(-1)))

    if not calculate_accuracy:
        return air_attenuation

    # calculate accuracy
    accuracy = np.zeros_like(air_attenuation) - 1
    freq2pressure = frequency/atmospheric_pressure
    vapor_1_mask = 0.05 <= p_vapor <= 5
    vapor_2_mask = not vapor_1_mask and 0.005 <= p_vapor
    vapor_3_mask = 0.005 > p_vapor
    temp_1_mask = -20 <= temperature <= 50
    temp_2_mask = -70 <= temperature and not temp_1_mask
    atm_mask = atmospheric_pressure <= 200000
    freq2pressure_mask = 4e-4 <= freq2pressure <= 10
    common_mask = atm_mask and freq2pressure_mask
    accuracy_10 = vapor_1_mask and temp_1_mask and common_mask
    accuracy_20 = vapor_2_mask and temp_1_mask and common_mask
    accuracy_50 = vapor_3_mask and temp_2_mask and common_mask
    accuracy[accuracy_10] = 10
    accuracy[accuracy_20] = 20
    accuracy[accuracy_50] = 50

    return air_attenuation, accuracy


def _p_sat_water(temperature):
    """Calculate the Water Saturation Pressure.

    Parameters
    ----------
    temperature : float, array_like
        Temperature in degree Celsius.

    Returns
    -------
    p_sat : float, array_like
        Water Saturation Pressure
    """
    temperature = np.atleast_1d(temperature)
    p_sat = np.atleast_1d(np.zeros_like(temperature))
    mask_temp = temperature < 0
    mask_temp_neg = temperature >= 0
    p_sat[mask_temp] = 6.1115*np.exp((
        23.036-temperature[mask_temp]/333.7)*(
            temperature[mask_temp]/(279.82+temperature[mask_temp])))
    p_sat[mask_temp_neg] = 6.1121*np.exp((
        18.678-temperature[mask_temp_neg]/234.5)*(
            temperature[mask_temp_neg]/(257.14+temperature[mask_temp_neg])))
    print(p_sat)
    if isinstance(temperature, np.ndarray):
        return p_sat
    else:
        return p_sat[0]

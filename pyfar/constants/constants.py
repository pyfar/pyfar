"""Constant calculation."""
import numpy as np
import pyfar as pf
from typing import Literal
import warnings


def air_attenuation(
        temperature, frequencies, relative_humidity,
        atmospheric_pressure=None):
    r"""Calculate the pure tone attenuation of sound in air according to
    ISO 9613-1.

    Calculation is in accordance with ISO 9613-1 [#]_. The shape of the
    outputs is broadcasted from the shapes of the ``temperature``,
    ``relative_humidity``, and ``atmospheric_pressure``.
    The frequency bins represents the last dimension.

    Parameters
    ----------
    temperature : float, array_like
        Temperature in degree Celsius.
        Must be in the range of -20°C to 50°C for accuracy of +/-10% or
        must be greater than -70°C for accuracy of +/-50%.
    frequencies : float, array_like
        Frequency in Hz. Must be greater than 50 Hz.
        Just one dimensional array is allowed.
    relative_humidity : float, array_like
        Relative humidity in the range from 0 to 1.
    atmospheric_pressure : float, array_like, optional
        Atmospheric pressure in Pascal, by default
        :py:attr:`reference_atmospheric_pressure`.

    Returns
    -------
    alpha : np.ndarray[float]
        Pure tone air attenuation coefficient in decibels per meter for
        atmospheric absorption.
    m : :py:class:`~pyfar.FrequencyData`
        Pure tone air attenuation coefficient per meter for
        atmospheric absorption. The parameter ``m`` is calculated as
        :math:`m = \alpha / (10 \cdot \log_{10}(e))`.
    accuracy : :py:class:`~pyfar.FrequencyData`
        accuracy of the results according to the standard:

        ``10``, +/- 10% accuracy
            - molar concentration of water vapour: 0.05% to 5 %.
            - air temperature: 253.15 K to 323.15 (-20 °C to +50°C)
            - atmospheric pressure: less than 200 000 Pa (2 atm)
            - frequency-to-pressure ratio: 0.0004 Hz/Pa to 10 Hz/Pa.

        ``20``, +/- 20% accuracy
            - molar concentration of water vapour: 0.005 % to 0.05 %,
              and greater than 5%
            - air temperature: 253.15 K to 323.15 (-20 °C to +50°C)
            - atmospheric pressure: less than 200 000 Pa (2 atm)
            - frequency-to-pressure ratio: 0.0004 Hz/Pa to 10 Hz/Pa.

        ``50``, +/- 50% accuracy
            - molar concentration of water vapour: less than 0.005%
            - air temperature: greater than 200 K (- 73 °C)
            - atmospheric pressure: less than 200 000 Pa (2 atm)
            - frequency-to-pressure ratio: 0.0004 Hz/Pa to 10 Hz/Pa.

        ``-1``, no valid result
            else.

    References
    ----------
    .. [#] ISO 9613-1:1993, Acoustics -- Attenuation of sound during
           propagation outdoors -- Part 1: Calculation of the absorption of
           sound by the atmosphere.
    """
    if atmospheric_pressure is None:
        atmospheric_pressure = pf.constants.reference_atmospheric_pressure
    # check inputs
    if not isinstance(temperature, (int, float, np.ndarray, list, tuple)):
        raise TypeError(
            'temperature must be a number or array of numbers')
    if not isinstance(frequencies, (int, float, np.ndarray, list, tuple)):
        raise TypeError(
            'frequencies must be a number or array of numbers')
    if not isinstance(
            relative_humidity, (int, float, np.ndarray, list, tuple)):
        raise TypeError(
            'relative_humidity must be a number or array of numbers')
    if np.array(frequencies).ndim > 1:
        raise ValueError('frequencies must be one dimensional.')
    if not isinstance(
            atmospheric_pressure, (int, float, np.ndarray, list, tuple)):
        raise TypeError(
            'atmospheric_pressure must be a number or array of numbers')

    # check if broadcastable
    try:
        _ = np.broadcast_shapes(
            np.atleast_1d(temperature).shape,
            np.atleast_1d(relative_humidity).shape,
            np.atleast_1d(atmospheric_pressure).shape)
    except ValueError as e:
        raise ValueError(
            'temperature, relative_humidity, and atmospheric_pressure must '
            'have the same shape or be broadcastable.') from e

    # check limits
    if np.any(np.array(temperature) < -73):
        raise ValueError("Temperature must be greater than -73°C.")
    if np.any(np.array(frequencies) < 50):
        raise ValueError("frequencies must be greater than 50 Hz.")
    if np.any(np.array(relative_humidity) < 0) or np.any(
            np.array(relative_humidity) > 1):
        raise ValueError("Relative humidity must be between 0 and 1.")
    if np.any(np.array(atmospheric_pressure) > 200000):
        raise ValueError("Atmospheric pressure must be less than 200 kPa.")
    # convert arrays
    temperature = np.array(
        temperature, dtype=float)[..., np.newaxis]
    relative_humidity = np.array(
        relative_humidity, dtype=float)[..., np.newaxis]
    atmospheric_pressure = np.array(
        atmospheric_pressure, dtype=float)[..., np.newaxis]
    frequencies = np.array(frequencies, dtype=float)

    # calculate air attenuation
    p_atmospheric_ref = pf.constants.reference_atmospheric_pressure
    t_degree_ref = pf.constants.reference_air_temperature_celsius

    h_r = relative_humidity*100
    p_a = atmospheric_pressure
    p_r = p_atmospheric_ref
    f = frequencies
    T = temperature - pf.constants.absolute_zero_celsius
    T_0 = t_degree_ref - pf.constants.absolute_zero_celsius

    # saturation vapour pressure
    p_sat = _saturation_vapour_pressure_iso(temperature)

    # molar concentration of water vapor as a percentage (Equation B.1)
    h = h_r * (p_sat / p_r) * (p_a / p_r)

    # Oxygen relaxation frequency (Eq. 3)
    f_rO = (p_a/p_r)*(24+4.04e4*h*(0.02+h)/(0.391+h))

    # Nitrogen relaxation frequency (Eq. 4)
    f_rN = (p_a/p_r)*(T/T_0)**(-1/2)*(9+280*h*np.exp(
        -4.17*((T/T_0)**(-1/3)-1)))

    # air attenuation (Eq. 5)
    alpha = 8.686*f**2*((1.84e-11*p_r/p_a*(T/T_0)**(1/2)) + \
        (T/T_0)**(-5/2)*(0.01275*np.exp(-2239.1/T)*(f_rO + (f**2/f_rO))**(-1)
        +0.1068*np.exp(-3352/T) * (f_rN + (f**2/f_rN))**(-1)))

    # Equation 3: ISO 17497-1:2004
    m = pf.FrequencyData(
        alpha / (10*np.log10(np.exp(1))), frequencies=frequencies)

    # calculate accuracy
    accuracy = _air_attenuation_accuracy(
        h, temperature, atmospheric_pressure, frequencies, m.freq.shape)

    return alpha, m, accuracy


def _air_attenuation_accuracy(
        concentration_water_vapour, temperature, atmospheric_pressure,
        frequencies, shape):
    """Calculate the accuracy of the air attenuation calculation.

    Calculation is in accordance with ISO 9613-1 [#]_. This method is used in
    :py:func:`~pyfar.constants.air_attenuation` to calculate the accuracy of
    the air attenuation calculation.

    Parameters
    ----------
    concentration_water_vapour : float, array_like
        Molar concentration of water vapor as a percentage.
        Must be between 0% and 100%.
    temperature : float, array_like
        Temperature in degree Celsius.
        Must be above -273.15°C.
    atmospheric_pressure : float, array_like
        Atmospheric pressure in Pascal.
        Must be above 0Pa.
    frequencies : float, array_like
        Frequency in Hz.
        Must be larger than 0 Hz.
    shape : tuple
        Shape of the output.

    Returns
    -------
    accuracy : :py:class:`~pyfar.classes.FrequencyData`
        accuracy of the results according to the standard:

            ``10``, +/- 10% accuracy
                - molar concentration of water vapour: 0.05% to 5 %.
                - air temperature: 253.15 K to 323.15 (-20 °C to +50°C)
                - atmospheric pressure: less than 200 000 Pa (2 atm)
                - frequency-to-pressure ratio: 4 x 10-4 Hz/Pa to 10 Hz/Pa.

            ``20``, +/- 20% accuracy
                - molar concentration of water vapour: 0.005 % to 0.05 %,
                  and greater than 5%
                - air temperature: 253.15 K to 323.15 (-20 °C to +50°C)
                - atmospheric pressure: less than 200 000 Pa (2 atm)
                - frequency-to-pressure ratio: 4 x 10-4 Hz/Pa to 10 Hz/Pa.

            ``50``, +/- 50% accuracy
                - molar concentration of water vapour: less than 0.005%
                - air temperature: greater than 200 K (- 73 °C)
                - atmospheric pressure: less than 200 000 Pa (2 atm)
                - frequency-to-pressure ratio: 4 x 10-4 Hz/Pa to 10 Hz/Pa.

            ``-1``, no valid result
                else.

    References
    ----------
    .. [#] ISO 9613-1:1993, Acoustics -- Attenuation of sound during
           propagation outdoors -- Part 1: Calculation of the absorption of
           sound by the atmosphere.
    """
    if np.any(np.array(concentration_water_vapour) < 0) or np.any(
            np.array(concentration_water_vapour) > 100):
        raise ValueError(
            r"Concentration of water vapour must be between 0% and 100%.")
    if np.any(np.array(temperature) < -273.15):
        raise ValueError(
            "Temperature must be greater than -273.15°C.")
    if np.any(np.array(atmospheric_pressure) < 0):
        raise ValueError(
            "Atmospheric pressure must be greater than 0 Pa.")
    if np.any(np.array(frequencies) < 0):
        raise ValueError(
            "Frequencies must be positive.")

    # broadcast inputs
    atmospheric_pressure = np.broadcast_to(atmospheric_pressure, shape)
    h_water_vapor = np.broadcast_to(concentration_water_vapour, shape)
    frequency_pressure_ratio = frequencies/atmospheric_pressure
    accuracy = np.zeros(shape) - 1

    # atmospheric pressure < 200 kPa
    atm_mask = atmospheric_pressure < 200000
    # frequency-to-pressure ratio: 4 x 10-4 Hz/Pa to 10 Hz/Pa
    frequency_pressure_ratio_mask = (4e-4 <= frequency_pressure_ratio) & (
        frequency_pressure_ratio <= 10)
    common_mask = atm_mask & frequency_pressure_ratio_mask

    # molar concentration of water vapour: 0.05% to 5 %
    vapor_10_mask = (0.05 <= h_water_vapor) & (h_water_vapor <= 5)
    # molar concentration of water vapour: 0.005% to 0.05 % and greater than 5%
    vapor_20_mask = (5 < h_water_vapor) | (
        (0.005 <= h_water_vapor) & (h_water_vapor < 0.05))
    # molar concentration of water vapour: less than 0.005%
    vapor_50_mask = (0.005 > h_water_vapor)

    # air temperature: 253,15 K to 323,15 (-20 °C to +50°C)
    temp_20_mask = (-20 <= temperature) & (temperature <= 50)
    # air temperature: greater than 200 K (- 73 °C)
    temp_50_mask = (-73 <= temperature)

    # apply masks
    accuracy_50 = common_mask & temp_50_mask & (
        vapor_10_mask | vapor_20_mask | vapor_50_mask)
    accuracy[accuracy_50] = 50

    accuracy_20 = common_mask & temp_20_mask & (
        vapor_10_mask | vapor_20_mask)
    accuracy[accuracy_20] = 20

    accuracy_10 = vapor_10_mask & common_mask & temp_20_mask
    accuracy[accuracy_10] = 10

    # return FrequencyData object
    return pf.FrequencyData(accuracy, frequencies=frequencies)


def _saturation_vapour_pressure_iso(temperature):
    """Calculates the saturation vapour pressure after ISO 9613-1:1993.

    This method is used in the :py:func:`air_attenuation` function to calculate
    the saturation vapour pressure of water vapour in air as a close
    approximation to
    those calculated by the World Meteorological Organization.
    The method is described in Equation B.2 and B.3 in [#]_.

    Parameters
    ----------
    temperature : float, array_like
        Temperature in degree Celsius.
        Must be in the range of -20°C to 50°C for accuracy of +/-10% or
        must be greater than -70°C for accuracy of +/-50%.

    Returns
    -------
    p_sat : float, array_like
        Saturation vapour pressure in Pascal.

    References
    ----------
    .. [#] ISO 9613-1:1993, Acoustics -- Attenuation of sound during
           propagation outdoors -- Part 1: Calculation of the absorption of
           sound by the atmosphere.
    """
    T = temperature - pf.constants.absolute_zero_celsius
    p_r = pf.constants.reference_atmospheric_pressure
    # triple-point isotherm temperature of 273.16 K
    T_01 = 273.16

    # saturation vapour pressure (Equation B.2 and B.3)
    C = -6.8346*(T_01/T)**1.261+4.6151
    return 10**C * p_r


def saturation_vapor_pressure_magnus(temperature):
    r"""
    Calculate the saturation vapor pressure of water in Pascal using the
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

    # Eq. (21)
    e_s = 6.1094 * np.exp((17.625 * temperature) / (temperature + 243.04))
    return 100 * e_s


def density_of_air(
        temperature, relative_humidity, atmospheric_pressure=None,
        saturation_vapor_pressure=None):
    r"""
    Calculate the density of air in kg/m³ based on the temperature,
    relative humidity, and atmospheric pressure.

    The density of air is calculated based on chapter 6.3 in [#]_.
    All input parameters must be broadcastable to the same shape.

    Parameters
    ----------
    temperature : float, array_like
        Temperature in degrees Celsius (°C).
    relative_humidity : float, array_like
        Relative humidity in the range from 0 to 1.
    atmospheric_pressure : float, array_like, optional
        Atmospheric pressure in Pascal (Pa), by default
        :py:attr:`reference_atmospheric_pressure`.
    saturation_vapor_pressure : float, array_like, optional
        Saturation vapor pressure in Pascal (Pa).
        The default uses the value and valid temperature range from
        :py:func:`~pyfar.constants.saturation_vapor_pressure_magnus`.

    Returns
    -------
    density : float, array_like
        Density of air in kg/m³.

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
    if np.any(np.array(relative_humidity) < 0) or np.any(
            np.array(relative_humidity) > 1):
        raise ValueError("Relative humidity must be between 0 and 1.")
    if np.any(np.array(atmospheric_pressure) <= 0):
        raise ValueError("Atmospheric pressure must be larger than 0 Pa.")

    P = np.array(atmospheric_pressure, dtype=float)  # Pa
    relative_humidity = np.array(relative_humidity, dtype=float)

    # Constants according to 6.3.2
    R = 8.314  # J/(K mol)
    mu_a = 28.97*1e-3  # kg/mol
    mu_w = 18.02*1e-3  # kg/mol

    # next to Equation 6.69
    R_a = R / mu_a
    alpha = mu_a / mu_w

    # partial pressure of water vapor in Pa
    if saturation_vapor_pressure is None:
        p = saturation_vapor_pressure_magnus(temperature)  # Pa
    else:
        p = np.array(saturation_vapor_pressure, dtype=float)  # Pa
    e = relative_humidity * p  # Pa

    # Equation 6.70
    C = (e/P) / (alpha * (1 - e/P))

    # Equation 6.71
    return P / (R_a * (temperature + 273.15)) * (1+C)/(1+alpha*C)


def fractional_octave_filter_tolerance(
        exact_center_frequency: float,
        num_fractions: Literal[1, 3],
        tolerance_class : Literal[1, 2]):
    r"""
    Calculate the tolerance limits for fractional octave band filters.

    Calculation is in accordance with IEC 61260-1:2014 [#]_ (Section 5.10 and
    Table 1).

    .. note ::
        The standard defines some lower tolerance limits as :math:`-\infty`,
        which is inconvenient for plotting. The returned tolerance is -60000 dB
        in these cases, which is below the smallest possible value of
        ``20*np.log10(np.finfo(float).tiny``
        :math:`\approx`-6000 dB.

    Parameters
    ----------
    exact_center_frequency : float
        The exact center frequency of the band filter in Hz (see
        :py:func:`~pyfar.dsp.filter.fractional_octave_frequencies`).
    num_fractions : Literal[1, 3]
        The number of bands an octave is divided into. ``1`` for octave bands
        and ``3`` for third octave bands.
    tolerance_class : Literal[1, 2]
        The tolerance class as defined in the standard. Must be ``1`` or ``2``.

    Returns
    -------
    lower_tolerance : numpy array
        Lower tolerance limits in dB of shape (19, ).
    upper_tolerance : numpy array
        Upper tolerance limits in dB of shape (19, ).
    frequencies : numpy array
        The frequencies in Hz at which the tolerance is given of shape (19, ).

    References
    ----------
    .. [#] IEC61260-1:2014 Octave-band and fractional-octave-band filters.
           Part 1: Specifications.

    Examples
    --------
    Class 1 tolerance region and filter for the 1000 Hz octave band.

    .. plot::

        >>> import pyfar as pf
        >>> import matplotlib.pyplot as plt
        >>>
        >>> lower, upper, frequencies = \
        ...     pf.constants.fractional_octave_filter_tolerance(
        ...         exact_center_frequency=1000, num_fractions=1,
        ...         tolerance_class=1)
        >>>
        >>> octave_filter = pf.dsp.filter.fractional_octave_bands(
        ...     pf.signals.impulse(2**12), num_fractions=1,
        ...     frequency_range=(1000, 1000))
        >>>
        >>> ax = pf.plot.freq(octave_filter, color='k', label='Octave filter')
        >>> plt.fill_between(
        ...     frequencies, lower, upper,
        ...     facecolor='g', alpha=.25, label='Class 1 Tolerance')
        >>> ax.set_ylim(-70, 5)
        >>> ax.set_xlim(63, 15_850)
        >>> ax.legend()
    """

    if not isinstance(exact_center_frequency, (int, float)):
        raise ValueError('The exact center frequency must be a float '
                         'or integer number')
    if num_fractions not in [1, 3]:
        raise ValueError(
            f"num_fractions is {num_fractions} but must be 1 or 3")
    if tolerance_class not in [1, 2]:
        raise ValueError(
            f"tolerance_class is {tolerance_class} but must be 1 or 2")

    # constants from DIN EN 61260-1:2014-10, Sec. 5.10 and Tab. 1
    # (define only first half due to symmetry)
    G = 10**(3/10)  # octave ratio according to Eq. (1)
    freq_offset = 1e-6  # small frequency offset to model discontinuities in
                        # tolerance curve
    min_dB = -60e3      # dB value to be used instead of minus infinity
                        # (number required for plots)
    relative_frequencies = [-4, -3, -2, -1, -0.5-freq_offset, -0.5+freq_offset,
                            -3/8, -1/4, -1/8, 0]
    if tolerance_class == 1:
        upper_tolerance = [
            -70, -60, -40.5, -16.6, -1.2, 0.4, 0.4, 0.4, 0.4, 0.4]
        lower_tolerance = [min_dB, min_dB, min_dB, min_dB, min_dB, -5.3,
                           -1.4, -0.7, -0.5, -0.4]
    else:
        upper_tolerance = [
            -60, -54, -39.5, -15.6, -0.8, 0.6, 0.6, 0.6, 0.6, 0.6]
        lower_tolerance = [min_dB, min_dB, min_dB, min_dB, min_dB, -5.8,
                           -1.7, -0.9, -0.7, -0.6]

    relative_frequencies = np.hstack((
        relative_frequencies, -np.flip(relative_frequencies[:-1])))
    upper_tolerance = np.hstack((
        upper_tolerance, np.flip(upper_tolerance[:-1])))
    lower_tolerance = np.hstack((
        lower_tolerance, np.flip(lower_tolerance[:-1])))

    # scale frequencies to bandwidth
    relative_frequencies /= num_fractions

    frequencies = exact_center_frequency * G**relative_frequencies

    return lower_tolerance, upper_tolerance, frequencies

def fractional_octave_frequencies_nominal(num_fractions=1,
                                          frequency_range=(20, 20e3)):
    """Return the nominal center frequencies for octave-band and
    one-third-octave-band filters.

    Nominal center frequencies, as specified in the IEC 61260-1:2014 standard
    [#]_ (Section 5.5 and Annex E), are standardized values that approximate
    the exact center frequencies. They are defined from 10 Hz to 20 kHz.

    Parameters
    ----------
    num_fractions : int, optional
        The number of octave fractions. ``1`` returns octave center
        frequencies, ``3`` returns third-octave center frequencies.
        The default is ``1``.
    frequency_range : array, tuple
        The lower and upper frequency limits, the default is
        ``(20, 20e3)`` following IEC 61260-1 [#]_.
        E.g. ``(10, 20e3)`` would follow IEC 61672-1.

    Returns
    -------
    nominal : numpy.ndarray of float
        The nominal center frequencies.

    References
    ----------
    .. [#] International Electrotechnical Commission, "IEC 61260-1:2014 -
        Electroacoustics - Octave-band and fractional-octave-band filters -
        Part 1: Specifications", IEC, 2014.

    .. [#] International Electrotechnical Commission,
        "IEC 61672-1:2013 - Electroacoustics - Sound level meters - Part 1:
        Specifications", IEC, 2013.
    """
    # IEC 61260-1 Eq. (1)
    G = 10**(3/10)
    f_lims = np.asarray(frequency_range)
    if f_lims.size != 2:
        raise ValueError(
            "You need to specify a lower and upper limit frequency.")
    if f_lims[0] > f_lims[1]:
        raise ValueError(
            "The second frequency needs to be higher than the first.")

    if num_fractions == 1:
        if (f_lims[0] < 15.8*G**(-1/2)) or (f_lims[1] >
                                             15848.93192*G**(1/2)):
            warnings.warn(
                "The nominal center frequencies for octave-band " \
                "are defined only from 11.2 Hz to 22387.2 Hz.", UserWarning,
                stacklevel=2)
        nominal = np.array([
                16, 31.5, 63, 125, 250, 500, 1e3,
                2e3, 4e3, 8e3, 16e3], dtype=float)
    elif num_fractions == 3:
        if (f_lims[0] < 10*G**(-1/6)) or (f_lims[1] > 19952.62315*G**(1/6)):
            warnings.warn(
                "The nominal center frequencies for one-third-octave-band " \
                "are defined only from 8.91 Hz to 22387.2 Hz.", UserWarning,
                stacklevel=2)
        nominal = np.array([
                10, 12.5, 16, 20, 25, 31.5, 40, 50, 63, 80,
                100, 125, 160, 200, 250, 315, 400, 500, 630,
                800, 1000, 1250, 1600, 2000, 2500, 3150, 4000,
                5000, 6300, 8000, 10000, 12500, 16000, 20000], dtype=float)
    else:
        raise ValueError('num_fractions must be 1 or 3')

    mask = (nominal >= f_lims[0]) & (nominal <= f_lims[1])
    nominal = nominal[mask]
    return nominal

def fractional_octave_frequencies_exact(
        num_fractions=1, frequency_range=(20, 20e3)):
    r"""Return the exact center and cutoff frequencies for
    fractional-octave-band filters.

    The frequencies are calculated in accordance with the IEC 61260-1:2014
    standard [#]_ (Sections 5.2, 5.3, 5.4 and 5.6).

    The octave frequency ratio, :math:`G`, is given by the following
    expression.

    .. math::

        G = 10^{\tfrac{3}{10}}

    The center frequencies :math:`f_m` are calculated using formula
    :eq:`eq_center_odd` for odd values of :math:`b` and formula
    :eq:`eq_center_even` for even values of :math:`b`.

    .. math::
        :label: eq_center_odd

        f_m = f_r \cdot G^{ \tfrac{x}{b}}

    .. math::
        :label: eq_center_even

        f_m = f_r \cdot G^{ \tfrac{2x+1}{2b}}

    where:

    - :math:`b` is the number of octave fractions.
    - :math:`f_r` is the reference frequency, set to 1000 Hz.
    - :math:`x` is the index of the frequency band.

    Parameters
    ----------
    num_fractions : int, optional
        The number of bands an octave is divided into. E.g., ``1`` refers to
        octave bands and ``3`` to third octave bands. The default is ``1``.
        All positive integers are allowed.
    frequency_range : array, tuple
        The lower and upper frequency limits, the default is
        ``(20, 20e3)``.

    Returns
    -------
    center_frequencies : numpy.ndarray
        The exact center frequencies in Hz, resulting in a uniform distribution
        of frequency bands over the frequency range.
    lower_cutoff_frequencies : numpy.ndarray
        The lower cutoff frequencies in Hz of the bandpass filters
        for each band.
    upper_cutoff_frequencies : numpy.ndarray
        The upper cutoff frequencies in Hz of the bandpass filters
        for each band.

    References
    ----------
    .. [#] International Electrotechnical Commission, "IEC 61260-1:2014 -
        Electroacoustics - Octave-band and fractional-octave-band filters -
        Part 1: Specifications", IEC, 2014.
    """
    # IEC 61260-1 Eq. (1)
    G = 10**(3/10)
    ref_freq = 1e3
    Nmax = np.around(num_fractions*(np.log2(frequency_range[1]/ref_freq)))
    Nmin = np.around(num_fractions*(np.log2(ref_freq/frequency_range[0])))

    indices = np.arange(-Nmin, Nmax+1)
    print(indices)
    if num_fractions % 2 != 0:
        # IEC 61260-1 Eq. (2)
        center_frequencies = ref_freq * (G)**(indices / num_fractions)
    else:
        # IEC 61260-1 Eq. (3)
        center_frequencies = ref_freq * (G)**((2*indices + 1)/
                                              (2*num_fractions))

    # IEC 61260-1 Eq. (5)
    upper_cutoff_frequencies = center_frequencies * G**(1/2/num_fractions)
    # IEC 61260-1 Eq. (4)
    lower_cutoff_frequencies = center_frequencies * G**(-1/2/num_fractions)
    return (center_frequencies,
            lower_cutoff_frequencies, upper_cutoff_frequencies)

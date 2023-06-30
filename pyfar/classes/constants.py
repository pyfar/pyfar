import numpy as np
from dataclasses import dataclass


@dataclass
class Constants:
    temperature_degree: float
    temperature_kelvin: float = 0.0
    relative_humidity: float = 0.5
    T_kelvin_zero_c: float = 273.15
    atmospheric_pressure: float = 101325
    atmospheric_pressure_ref: float = 101325
    frequency: float = 1e3

    def __init__(
            self, temperature_degree=20, relative_humidity=0.5,
            atmospheric_pressure=101325, frequency=1e3):
        """Commonly used constants. So far only the speed of sound.
        Parameters
        ----------
        temperature_degree : float
            The temperature in degrees Celsius.
        relative_humidity : float, [0, 1]
            The relative air humidity in the interval [0, 1].
        atmospheric_pressure
            The atmospheric pressure in Pascals.
        frequency
            The frequency in Hertz.
        """
        self.temperature_degree = temperature_degree
        self.relative_humidity = relative_humidity
        self.atmospheric_pressure = atmospheric_pressure
        self.frequency = frequency

    @property
    def p_sat(self) -> float:
        """saturation vapor pressure
        """
        V = -6.8346*(
            self.temperature_degree_ref / self.temperature_kelvin)**1.261 \
                + 4.6151
        return self.atmospheric_pressure_ref*10**V

    @property
    def R_l(self) -> float:
        """gas constant for dry air [J/(kg*K)]"""
        # molar mass of dry air
        M_r = 0.0289644

        # molar gas constant for air
        R_mol = 8.31

        # gas constant for dry air [J/(kg*K)]
        R_l = R_mol/M_r
        return R_l

    @property
    def R_f(self) -> float:
        # molar concentration of water vapor in percent
        h = 100*self.relative_humidity*self.p_sat/self.atmospheric_pressure

        # gas constant for air with relative humidity phi [J/(kg K)]
        R_f = self.R_l/(1-(h/100)*(1-self.R_l/self.R_d))
        return R_f

    @property
    def rho_0(self) -> float:
        # air density
        return self.atmospheric_pressure / (self.R_f*self.temperature_kelvin)


def constants2(
        temperature=20, humidity=0.5, atmospheric_pressure=101325, freq=1e3):
    """Commonly used constants. So far only the speed of sound.

    Parameters
    ----------
    temperature : float
        The temperature in degrees Celsius.
    humidity : float, [0, 1]
        The relative air humidity in the interval [0, 1].
    atmospheric_pressure
        The atmospheric pressure in Pascals.

    Returns
    -------
    constants : dict
        A dictionary with the corresponding constants.

    """

    T_kelvin_zero_c = 273.15
    temperature_degree = temperature
    rel_hum = humidity
    temperature_kelvin = temperature_degree + T_kelvin_zero_c
    temperature_degree_ref = 20 + T_kelvin_zero_c
    atmospheric_pressure_ref = 101325

    V = -6.8346*(temperature_degree_ref / temperature_kelvin)**1.261 + 4.6151

    # saturation vapor pressure
    p_sat = atmospheric_pressure_ref*10**V



    # gas constant of water vapor
    R_d = 461

    # gas constant for air with relative humidity phi [J/(kg K)]
    R_f = R_l/(1-(h/100)*(1-R_l/R_d))

    # heat capacity ratio
    kappa = 1.4

    # heat conductivity
    nu = 0.0261

    # specific heat capacity
    C_v = 718


    # air viscosity (at 273K)
    # eta = 17.1*1e-6
    eta = 1.485e-6*temperature_kelvin**(3/2)/(temperature_kelvin+110.4)

    # - reference pressure for SPL
    p_b = 2e-5

    # Prantel number
    Pr = 1e9/(1.1*temperature_degree**3 - 120*temperature_degree**2 + 322000*temperature_degree + 1.393e9)

    # speed of sound
    c = np.sqrt(kappa*R_f*temperature_kelvin)

    # ISO 9613-1 Eq 3, 4
    frO = (atmospheric_pressure/atmospheric_pressure_ref)*(24 + 4.04e4*h*(0.02 + h)/(0.391 + h))
    frN = (atmospheric_pressure/atmospheric_pressure_ref)*(temperature_kelvin/temperature_degree_ref)**(-1/2)*(9 + 280*h*np.exp(-4.17*((temperature_kelvin/temperature_degree_ref)**(-1/3) - 1)))

    # m (1/m), factor 2 comes from conversion Neper -> dB -> linear
    # for dB/m multiply by 10*log10(np.exp(1)) = 4.3429
    m = 2*freq**2*((1.84e-11*atmospheric_pressure_ref/atmospheric_pressure)*(temperature_kelvin/temperature_degree_ref)**(1/2)) + (temperature_kelvin/temperature_degree_ref)**(-5/2)*(
        0.01275*np.exp(-2239.1/temperature_kelvin)*frO/(frO**2 + freq**2) +
        0.1068*np.exp(-3352/temperature_kelvin)*frN/(frN**2+freq**2))

    return {
        'c': c, 'rho_0': rho_0, 'm': m, 'Pr': Pr, 'eta': eta, 'kappa': kappa}


def constants(
        temperature=20, humidity=0.5, atmospheric_pressure=101325, freq=1e3):
    """Commonly used constants. So far only the speed of sound.

    Parameters
    ----------
    temperature : float
        The temperature in degrees Celsius.
    humidity : float, [0, 1]
        The relative air humidity in the interval [0, 1].
    atmospheric_pressure
        The atmospheric pressure in Pascals.

    Returns
    -------
    constants : dict
        A dictionary with the corresponding constants.

    """

    T_kelvin_zero_c = 273.15
    T_c = temperature
    rel_hum = humidity
    T = T_c + T_kelvin_zero_c
    T_ref = 20 + T_kelvin_zero_c
    p_ref = 101325
    p_stat = atmospheric_pressure

    V = -6.8346*(T_ref / T)**1.261 + 4.6151

    # saturation vapor pressure
    p_sat = p_ref*10**V

    # molar concentration of water vapor in percent
    h = 100*rel_hum*p_sat/p_stat

    # molar mass of dry air
    M_r = 0.0289644

    # molar gas constant for air
    R_mol = 8.3144621

    # gas constant for dry air [J/(kg*K)]
    R_l = R_mol/M_r

    # gas constant of water vapor
    R_d = 461

    # gas constant for air with relative humidity phi [J/(kg K)]
    R_f = R_l/(1-(h/100)*(1-R_l/R_d))

    # heat capacity ratio
    kappa = 1.4

    # heat conductivity
    nu = 0.0261

    # specific heat capacity
    C_v = 718

    # air density
    rho_0 = p_stat / (R_f*T)

    # air viscosity (at 273K)
    eta = 17.1*1e-6

    # - reference pressure for SPL
    p_b = 2e-5

    # speed of sound
    c = np.sqrt(kappa*R_f*T)

    # ISO 9613-1 Eq 3, 4
    frO = (p_stat/p_ref)*(24 + 4.04e4*h*(0.02 + h)/(0.391 + h))
    frN = (p_stat/p_ref)*(T/T_ref)**(-1/2)*(9 + 280*h*np.exp(-4.17*((T/T_ref)**(-1/3) - 1)))

    # m (1/m), factor 2 comes from conversion Neper -> dB -> linear
    # for dB/m multiply by 10*log10(np.exp(1)) = 4.3429
    m = 2*freq**2*((1.84e-11*p_ref/p_stat)*(T/T_ref)**(1/2)) + (T/T_ref)**(-5/2)*(
        0.01275*np.exp(-2239.1/T)*frO/(frO**2 + freq**2) +
        0.1068*np.exp(-3352/(T))*frN/(frN**2+freq**2))

    consts = {'c': c, 'rho_0': rho_0, 'm': m}
    return consts


def speed_of_sound(temperature=20, humidity=0.5):
    """Calculate the speed of sound for a given temperature and air humidity.

    Parameters
    ----------
    temperature : float
        The temperature in degrees Celsius.
    humidity : float, [0, 1]
        The relative air humidity in the interval [0, 1].

    Returns
    -------
    speed_of_sound : float
        The speed of sound in meters per second.

    """
    const = constants(temperature=temperature, humidity=humidity)

    return const['c']


def speed_of_sound_cramer(
        temperature=20, humidity=0.5, atmospheric_pressure=101325, freq=1e3):

    T_kelvin_zero_c = 273.15
    T_c = temperature
    rel_hum = humidity
    T = T_c + T_kelvin_zero_c
    T_ref = 20 + T_kelvin_zero_c
    p_ref = 101325
    p_stat = atmospheric_pressure

    # V = -6.8346*(T_ref / T)**1.261 + 4.6151

    fw = 1.00519
    V = fw * 6.112*100 * np.exp((17.62*T_c) / (243.12+T_c))

    # saturation vapor pressure
    p_sat = p_ref*10**V

    # molar concentration of water vapor in percent
    h = 100*rel_hum*p_sat/p_stat

    # molar mass of dry air
    M_r = 0.0289644

    # molar gas constant for air
    R_mol = 8.31

    # gas constant for dry air [J/(kg*K)]
    R_l = R_mol/M_r

    # gas constant of water vapor
    R_d = 461

    # gas constant for air with relative humidity phi [J/(kg K)]
    R_f = R_l/(1-(h/100)*(1-R_l/R_d))

    # heat capacity ratio
    kappa = 1.4

    # heat conductivity
    nu = 0.0261

    # specific heat capacity
    C_v = 718

    # air density
    rho_0 = p_stat / (R_f*T)

    x_c = 0.314
    x_c = x_c / 100

    # % Correction for wet air (valid for -50 < T in Celsius < 90)
    fw = 1.00519

    # % saturation vapour pressure of water with correction for wet air
    # % according to: D. Sonntag, and D. Heinze (1982): Sättigungsdampfdruck-
    # % und Sättigungsdampfdichtetafeln für Wasser und Eis. (1. Aufl.),
    # % VEB Deutscher Verlag für Grundstoffindustrie
    # % (Magnus-Formel)
    # pws = fw * 611.213 * 10**((7.602*T_c) / (241.2+T_c))
    pws = fw * 6.112*100 * np.exp((17.62*T_c) / (243.12+T_c))

    # % mixing ratio (mole fraction) of water vapor
    xw = rel_hum * pws / p_stat

    # % Coefficients according to [2]
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

    # % approximation for c according to [2]
    c1 = a0+a1*T_c+a2*T_c**2
    c2 = (a3+a4*T_c+a5*T_c**2)*xw
    c3 = (a6+a7*T_c+a8*T_c**2)*p_stat
    c4 = (a9+a10*T_c+a11*T_c**2)*x_c
    c5 = a12*xw**2 + a13*p_stat**2 + a14*x_c**2 + a15*xw*p_stat*x_c

    return c1 + c2 + c3 + c4 + c5

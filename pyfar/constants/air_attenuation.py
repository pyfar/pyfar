import numpy as np


def air_attenuation_iso(
        temperature, frequency, relative_humidity,
        atmospheric_pressure=101325):


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

    return 8.686*f**2*((1.84e-11*p_r/p_a*(T/T_0)**(1/2)) + \
        (T/T_0)**(-5/2)*(0.01275*np.exp(-2239.1/T)*(f_rO + (f**2/f_rO))**(-1)
        +0.1068*np.exp(-3352/T) * (f_rN + (f**2/f_rN))**(-1)))


def _p_sat_water(temperature):
    if (temperature < 0):
        return 6.1115*np.exp((
            23.036-temperature/333.7)*(temperature/(279.82+temperature)))
    else:
        return 6.1121*np.exp((
            18.678-temperature/234.5)*(temperature/(257.14+temperature)))

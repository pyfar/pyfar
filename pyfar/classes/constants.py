import numpy as np
from dataclasses import dataclass, field


@dataclass
class Constants:
    """
    Dataclass for commonly used constants.

    Parameters
    ----------
    t_degree : float, optional
        The temperature in degrees Celsius. The default is 20 degree celsius.
    relative_humidity : float, [0, 1], optional
        The relative air humidity in the interval [0, 1]. The default is 0.5.
    frequency: float, optional
        The frequency in Hertz. The default is 1000 Hz.
    p_atmospheric : float, optional
        The atmospheric pressure in Pascals. The default is 101325 Pa.

    References
    ----------
    ..[1] Buck (1996), Buck Research CR-1A User's Manual, Appendix 1.    
    ..[2] NIST Chemistry WebBook, SRD 69
    ..[3] Gatley, Donald & Herrmann, Sebastian & Kretzschmar, Hans-Joachim, 
      (2008), A Twenty-First Century Molar Mass for Dry Air,
      HVAC&R Research, 14, 655-662, 10.1080/10789669.2008.10391032. 
    ..[4] tec-science (2020-03-25), "Viscosity of liquids and gases",
      https://www.tec-science.com/mechanics/gases-and-liquids/
      viscosity-of-liquids-and-gases/
    ..[5] https://www.tec-science.com/de/mechanik/gase-und-fluessigkeiten/
      prandtl-zahl/
    ..[6] ISO 9613-1 Eq 3-5
    ..[7] Cramer, Owen, The variation of the specific heat ratio and the speed 
      of sound in air with temperature, pressure, humidity, and CO2 
      concentration. J. Acoust. Soc. Am. 1 May 1993; 93 (5): 
      2510–2516. https://doi.org/10.1121/1.405827
    ..[8] Doiron, T. (2007), 20 Degrees Celsius-A Short History of the Standard 
      Reference Temperature for Industrial Dimensional Measurements, 
      Journal of Research (NIST JRES), National Institute of Standards and 
      Technology, Gaithersburg, MD, [online], 
      https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=823211
    """
    t_degree: float = 20
    relative_humidity: float = 0.5
    frequency: float = 1e3
    p_atmospheric: float = 101325
    p_atmospheric_ref: float = field(init=False)
    t_degree_ref: float = field(init=False)
    t_kelvin: float = field(init=False)
    t_kelvin_ref: float = field(init=False)

    def __post_init__(self):
        self.p_atmospheric_ref = 101325
        self.t_degree_ref = 20
        self.t_kelvin = (self.t_degree + 273.15)
        self.t_kelvin_ref = (self.t_degree_ref + 273.15)

    @property
    def p_sat_water(self) -> float:
        """
        Get saturation vapor pressure of water in [hPa] using the 
        Arden Buck equation. [1]_
        """
        if (self.t_degree < 0):
            return 6.1115*np.exp((23.036-self.t_degree/333.7)*
                                 (self.t_degree/
                                  (279.82+self.t_degree)))
        else:
            return 6.1121*np.exp((18.678-self.t_degree/234.5)*
                                 (self.t_degree/
                                  (257.14+self.t_degree)))


    @property
    def p_vapor(self) -> float:
        """
        Get vapor pressure of water for given relative humidity in [hPa].
        """
        return self.relative_humidity*self.p_sat_water
    
    @property
    def p_dry_air(self) -> float:
        """
        Get partial pressure of dry air in [hPa].
        """
        return self.p_atmospheric/100-self.p_vapor
    
    @property 
    def M_vapor(self) -> float:
        """
        Molar mass of water vapor in [kg/mol]. [2]_
        """
        return 0.0180153

    @property 
    def M_dry_air(self) -> float:
        """
        Molar mass of dry air in [kg/mol]. [3]_
        """
        return 0.028966
    
    @property
    def R(self) -> float:
        """
        Molar gas constant R in [J/(mol*K)], exact value since 2019 
        redifinition of SI base units.
        """
        return 8.31446261815324
    
    @property
    def R_vapor(self) -> float:
        """
        Get gas constant for water vapor in [J/(kg*K)].
        """
        return self.R/self.M_vapor

    @property
    def R_dry_air(self) -> float:
        """
        Get gas constant for dry air in [J/(kg*K)].
        """
        return self.R/self.M_dry_air
    
    @property
    def rho_humid_air(self) -> float:
        """
        Get density of humid air in [kg/m^3].
        """
        # Treating humid air as the mixture of two ideal gases
        return (self.p_dry_air*self.M_dry_air+self.p_vapor*self.M_vapor)*100/\
                    (self.R*self.t_kelvin)

    @property
    def eta(self) -> float:
        """
        Get dynamic viscosity of air in [Pa*s]. [4]_
        """
        # air viscosity (at 273K)
        # eta = 17.1*1e-6
        # return 1.485e-6*T**(3/2)/(T+110.4)
        return 2.791e-7*self.t_kelvin**(0.7355) 

    @property
    def c(self) -> float:
        """
        Get speed of sound in air in [m/s] using the ideal gas equation.
        """
        # adiabatic index for diatomic gases
        kappa = 1.4

        return np.sqrt(kappa*self.p_atmospheric/self.rho_humid_air)

    @property
    def Pr(self) -> float:
        """
        Get Prandtl number of air at an atmospheric pressure of 1 bar. [5]_
        """
        return 1e9/(1.1*self.t_degree**3-120*self.t_degree**2+ 
                    322000*self.t_degree+1.393e9)
    

    @property
    def m(self) -> float:
        """
        Get the attenuation coefficient in [dB/m] using the calculation method
        described in ISO 9613-1 Eq 3-5. [6]_ 
        """
        p_atm = self.p_atmospheric
        p_ref = self.p_atmospheric_ref
        f = self.frequency
        t_k = self.t_kelvin
        t_kr = self.t_kelvin_ref
        # molar concentration of water vapor as a percentage
        h = self.p_vapor/p_atm*10000

        # Oxygen relaxation frequency
        f_rO = (p_atm/p_ref)*(24+4.04e4*h*(0.02+h)/(0.391+h))
        # Nitrogen relaxation frequency
        f_rN = (p_atm/p_ref)*(t_k/t_kr)**(-1/2)*\
            (9+280*h*np.exp(-4.17*((t_k/t_kr)**(-1/3)-1)))

        return 8.686*f**2*((1.84e-11*p_ref/p_atm*(t_k/t_kr)**(1/2)) + \
            (t_k/t_kr)**(-5/2)*(0.01275*np.exp(-2239.1/t_k)*f_rO/(f_rO**2+f**2)
                                +0.1068*np.exp(-3352/t_k)*f_rN/(f_rN**2+f**2)))

    @property
    def c_cramer(self) -> float:
        """
        Get speed of sound using Cramers method described in [7]_.
        """
        rel_hum = self.relative_humidity
        p_stat = self.p_atmospheric
        T_c = self.t_degree

        x_c = 0.314
        x_c = x_c / 100

        # % Correction for wet air (valid for -50 < T in Celsius < 90)
        fw = 1.00519

        # saturation vapour pressure of water with correction for wet air
        # according to: D. Sonntag, and D. Heinze (1982): Sättigungsdampfdruck-
        # und Sättigungsdampfdichtetafeln für Wasser und Eis. (1. Aufl.),
        # VEB Deutscher Verlag für Grundstoffindustrie
        # (Magnus-Formel)
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

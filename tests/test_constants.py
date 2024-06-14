import pytest
from pyfar import Constants
import pyfar as pf
import numpy.testing as npt


def test_constants_init():
    """Test initialization of empty constants object."""
    const = Constants()
    assert isinstance(const, Constants)


@pytest.mark.parametrize('t_degree, p_sat', (
        [-40, 0.128412],
        [-20, 1.03239],
        [0, 6.11153],
        [20, 23.2932],
        [40, 73.8494],
        [60, 199.464],
        [80, 474.145]
))
def test_constants_p_sat_water(t_degree, p_sat):
    """
    Test calculation of saturation vapor pressure. Literature values found 
    in [1]_.

    References
    ----------
    ..[1] Huang, Jianhua. "A simple accurate formula for calculating saturation 
    vapor pressure of water and ice." Journal of Applied Meteorology and 
    Climatology 57.6 (2018): 1265-1272.
    """
    c = Constants(t_degree)
    npt.assert_allclose(c.p_sat_water, p_sat, rtol=0.01)


@pytest.mark.parametrize('t_degree, rho_air', (
        [-20, 1.394],
        [-10, 1.341],
        [0, 1.292],
        [10, 1.246],
        [20, 1.204],
        [30, 1.164]
))
def test_constants_rho_dry_air(t_degree, rho_air):
    """
    Test calculation of dry air density. Reference values from [1]_.
    
    References
    ----------
    ..[1] https://www.engineersedge.com/physics/viscosity_of_air_dynamic_and\
      _kinematic_14483.htm
    """
    c = Constants(t_degree,0)
    npt.assert_allclose(c.rho_humid_air, rho_air, rtol=0.01)


@pytest.mark.parametrize('t_degree, rho_air', (
        [-20, 1.376],
        [-10, 1.322],
        [0, 1.272],
        [10, 1.225],
        [20, 1.178],
        [30, 1.131],
        [40, 1.081],
        [50, 1.028]
))
def test_constants_rho_humid_air(t_degree, rho_air):
    """
    Test calculation of humid air density. Reference values from table 4 in 
    [1]_.
    
    References
    ----------
    ..[1] Hellmuth, O., Feistel, R. & Foken, T. Intercomparison of different 
      state-of-the-art formulations of the mass density of humid air. Bull. 
      of Atmos. Sci.& Technol. 2, 13 (2021). 
      https://doi.org/10.1007/s42865-021-00036-7
    """
    c = Constants(t_degree,1)
    npt.assert_allclose(c.rho_humid_air, rho_air, rtol=0.015)


@pytest.mark.parametrize('t_degree, dyn_viscosity', (
        [-20, 1.630e-5],
        [-10, 1.680e-5],
        [0, 1.729e-5],
        [10, 1.778e-5],
        [20, 1.825e-5],
        [30, 1.872e-5],
        [40, 1.918e-5],
        [50, 1.963e-5]
))
def test_constants_eta(t_degree, dyn_viscosity):
    """
    Test calculation of dynamic viscosity of air. Reference values from [1]_.

    References
    ----------
    ..[1] https://www.engineersedge.com/physics/viscosity_of_air_dynamic_and_
      kinematic_14483.htm
    """
    c = Constants(t_degree)
    npt.assert_allclose(c.eta, dyn_viscosity, rtol=0.01)

@pytest.mark.parametrize('t_degree, rh, p_atm, speed_of_sound', (
        [0, 0.0, 101325, 331.45],
        [0, 0.5, 101325, 331.6],
        [0, 1.0, 101325, 331.76],
        [0, 0.0, 80000, 331.45],
        [10, 0.0, 101325, 337.46],
        [10, 0.5, 101325, 337.78],
        [10, 1.0, 101325, 338.1],
        [10, 0.5, 80000, 337.86],
        [20, 0.0, 101325, 343.36],
        [20, 0.5, 101325, 343.99],
        [20, 1.0, 101325, 344.61],
        [20, 1.0, 80000, 344.94]
))
def test_constants_c(t_degree, rh, p_atm, speed_of_sound):
    """
    Test calculation of speed of sound in air. Reference values are 
    calculated using the NPL calculator [1]_.
    
    References
    ----------
    ..[1] http://resource.npl.co.uk/acoustics/techguides/speedair/
    """
    c = Constants(t_degree, rh, p_atmospheric=p_atm)
    npt.assert_allclose(c.c, speed_of_sound, rtol=0.01)


@pytest.mark.parametrize('t_degree, rh, p_atm, speed_of_sound', (
    [0, 0.0, 101325, 331.45],
    [0, 0.5, 101325, 331.6],
    [0, 1.0, 101325, 331.76],
    [0, 0.0, 80000, 331.45],
    [10, 0.0, 101325, 337.46],
    [10, 0.5, 101325, 337.78],
    [10, 1.0, 101325, 338.1],
    [10, 0.5, 80000, 337.86],
    [20, 0.0, 101325, 343.36],
    [20, 0.5, 101325, 343.99],
    [20, 1.0, 101325, 344.61],
    [20, 1.0, 80000, 344.94]
))
def test_constants_c_cramer(t_degree, rh, p_atm, speed_of_sound):
    """
    Test calculation of speed of sound in air for Cramers calculation method. 
    Reference values are calculated using the NPL calculator [1]_.
    
    References
    ----------
    ..[1] http://resource.npl.co.uk/acoustics/techguides/speedair/
    """
    c = Constants(t_degree, rh, p_atmospheric=p_atm)
    npt.assert_allclose(c.c_cramer, speed_of_sound, rtol=0.01)


@pytest.mark.parametrize('t_degree, prandtl_number', (
        [-33.2, 0.717],
        [-13.2, 0.713],
        [0, 0.711],
        [6.9, 0.710],
        [15.6, 0.709],
        [26.9, 0.707],
        [46.9, 0.705]
))
def test_constants_prandtl(t_degree, prandtl_number):
    """
    Test calculation of Prandtl number of air. Reference values from [1]_.

    References
    ----------
    ..[1] https://www.engineeringtoolbox.com/air-prandtl-number-viscosity-heat
      -capacity-thermal-conductivity-d_2009.html
    """
    c = Constants(t_degree)
    npt.assert_allclose(c.Pr, prandtl_number, rtol=0.01)

@pytest.mark.parametrize('t_degree, rh, f, attenuation_coefficient', (
    # [-15, 0.1, 125, 1.62],
    # [-15, 0.5, 1000, 13.2],
    # [-15, 1.0, 8000, 73.9],
    # [-15, 0.1, 8000, 12.0],
    [0, 0.1, 125, 1.30],
    [0, 0.5, 1000, 6.83],
    [0, 1.0, 8000, 129],
    [0, 0.5, 125, 0.411],
    [15, 0.1, 125, 0.735],
    [15, 0.5, 1000, 4.16],
    [15, 1.0, 8000, 68.1],
    [15, 1.0, 1000, 4.35]
))
def test_constants_attenuation(t_degree, rh, f, attenuation_coefficient):
    """
    Test calculation of the air attenuation coefficient. Reference values from 
    [1]_.

    Notes
    -----
    The calculation of the attenuation coefficient seems to be inaccurate for 
    low temperatures. Another check is required.

    References
    ----------
    ISO 9613-1 Eq 3-5
    """
    c = Constants(t_degree, rh, f)
    npt.assert_allclose(c.m*1000, attenuation_coefficient, rtol=0.02)
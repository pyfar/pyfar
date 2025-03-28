import pyfar as pf
import numpy as np

def test_reference_atmospheric_pressure():
    assert pf.constants.reference_atmospheric_pressure == 101325.0

def test_absolute_zero_celsius():
    assert pf.constants.absolute_zero_celsius == -273.15

def test_reference_sound_power():
    assert pf.constants.reference_sound_power == 1e-12

def test_reference_sound_pressure():
    assert pf.constants.reference_sound_pressure == 20e-6

def test_reference_air_temperature():
    assert pf.constants.reference_air_temperature_celsius == 20

def test_reference_speed_of_sound():
    assert pf.constants.reference_speed_of_sound == 343.2

def test_reference_air_density():
    assert pf.constants.reference_air_density == 1.204

def test_reference_air_impedance_approximation():
    assert np.round(pf.constants.reference_air_impedance, 1) == 413.2

def test_reference_air_impedance():
    c_ref = pf.constants.reference_speed_of_sound
    rho_atm = pf.constants.reference_air_density
    assert pf.constants.reference_air_impedance == c_ref * rho_atm

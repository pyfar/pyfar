import pyfar as pf

def test_standard_atmosphere_pressure():
    assert pf.constants.standard_atmosphere_pressure == 101325.0

def test_absolute_zero_celsius():
    assert pf.constants.absolute_zero_celsius == -273.15

def test_standard_air_density():
    assert pf.constants.standard_air_density == 1.204

def test_reference_sound_power():
    assert pf.constants.reference_sound_power == 1e-12

def test_reference_sound_pressure():
    assert pf.constants.reference_sound_pressure == 20e-6

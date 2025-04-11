import pytest
import numpy as np
import pyfar as pf
import numpy.testing as npt
import re


@pytest.mark.parametrize(("temperature", "relative_humidity", "expected"), [
    (0, 0.0047393, 331.5079),
    (0, 0.30174, 331.627),
    (0, 0.60032, 331.7063),
    (0, 0.99526, 331.9048),
    (10, 0.0063191, 337.4603),
    (10, 0.30332, 337.6984),
    (10, 0.6019, 337.8571),
    (10, 0.9921, 338.1746),
    (20, 0.0078989, 343.3333),
    (20, 0.30174, 343.7302),
    (20, 0.6019, 344.1667),
    (20, 0.98736, 344.6825),
    (30, 0.0047393, 349.1667),
    (30, 0.30332, 349.881),
    (30, 0.6019, 350.5556),
    (30, 0.98894, 351.5079),
    ])
def test_speed_of_sound_cramer_figure_1_rel_hum(
        temperature, relative_humidity, expected):
    """Test speed_of_sound_cramer with figure 1 vs relative humidity."""

    speed = pf.constants.speed_of_sound_cramer(
        temperature, relative_humidity, 314)
    npt.assert_almost_equal(speed, expected, 1)


@pytest.mark.parametrize((
        "temperature", "atmospheric_pressure", "expected"), [
            (0, 75265.49, 331.4732),
            (0, 85035.4, 331.3839),
            (0, 95070.8, 331.3839),
            (10, 75212.39, 337.4107),
            (10, 85088.5, 337.4554),
            (10, 95123.89, 337.5),
            (20, 75212.39, 343.3036),
            (20, 85088.5, 343.3482),
            (20, 95017.7, 343.3929),
            (30, 75053.1, 349.1000),
            (30, 85088.5, 349.1518),
            (30, 94858.41, 349.2411),
    ])
def test_speed_of_sound_cramer_figure_1_pressure(
        temperature, atmospheric_pressure, expected):
    """Test speed_of_sound_cramer with figure 1 vs atmospheric pressure."""
    relative_humidity = 0

    speed = pf.constants.speed_of_sound_cramer(
        temperature, relative_humidity, 314, atmospheric_pressure)
    npt.assert_almost_equal(speed, expected, 1)


@pytest.mark.parametrize((
        "temperature", "co2_percent", "expected"), [
            (0, 0.006079, 331.4818),
            (0, 0.29939, 331.2538),
            (0, 0.6003, 331.0258),
            (0, 0.9924, 330.6839),
            (10, 0.0075988, 337.4848),
            (10, 0.29939, 337.2188),
            (10, 0.60182, 336.9529),
            (10, 0.99088, 336.7249),
            (20, 0.0091185, 343.4498),
            (20, 0.30091, 343.1459),
            (20, 0.6003, 342.9559),
            (20, 0.98784, 342.576),
            (30, 0.0091185, 349.035),
            (30, 0.30091, 348.845),
            (30, 0.60182, 348.617),
            (30, 0.9848, 348.3511),
    ])
def test_speed_of_sound_cramer_figure_1_co2(
        temperature, co2_percent, expected):
    """Test speed_of_sound_cramer with figure 1 vs co2 concentration."""
    co2_ppm = co2_percent / 100 * 1e6  # in parts per million
    relative_humidity = 0

    speed = pf.constants.speed_of_sound_cramer(
        temperature, relative_humidity, co2_ppm=co2_ppm,
        )
    npt.assert_almost_equal(speed, expected, 1)


@pytest.mark.parametrize((
        "temperature", "relative_humidity", "atmospheric_pressure",
        "co2_ppm", "expected"), [
            (0, 0, 95070.8, 314, 331.3839),
            (0, 0, 95070.8, [314], 331.3839),
            (0, 0, [95070.8], [314], 331.3839),
            (0, [0], 95070.8, [314], 331.3839),
            ([0], 0, 95070.8, 314, 331.3839),
            ([0], 0, (95070.8, ), 314, 331.3839),
            ([0, 0], 0, 95070.8, 314, 331.3839),
            (np.array([[0, 0], [0, 0]]), 0, 95070.8, 314, 331.3839),
        ])
def test_speed_of_sound_cramer_array_like(
        temperature, relative_humidity, atmospheric_pressure,
        co2_ppm, expected):
    speed = pf.constants.speed_of_sound_cramer(
        temperature, relative_humidity, co2_ppm, atmospheric_pressure,
        )
    npt.assert_almost_equal(speed, expected, 1)


def test_speed_of_sound_cramer_invalid_temperature():
    match = 'Temperature must be between 0°C and 30°C.'
    with pytest.raises(ValueError, match=re.escape(match)):
        pf.constants.speed_of_sound_cramer(
            -10, .5)
    with pytest.raises(ValueError, match=re.escape(match)):
        pf.constants.speed_of_sound_cramer(
            31, .5)


def test_speed_of_sound_cramer_invalid_atmospheric_pressure():
    match = 'Atmospheric pressure must be between 75 000 Pa to 102 000 Pa.'
    with pytest.raises(ValueError, match=re.escape(match)):
        pf.constants.speed_of_sound_cramer(0, .5, 341, 74999)
    with pytest.raises(ValueError, match=re.escape(match)):
        pf.constants.speed_of_sound_cramer(1, .5, 341, 102001)


def test_speed_of_sound_cramer_invalid_relative_humidity():
    match = 'Relative humidity must be between 0 and 1.'
    with pytest.raises(ValueError, match=re.escape(match)):
        pf.constants.speed_of_sound_cramer(
            0, -0.1)
    with pytest.raises(ValueError, match=re.escape(match)):
        pf.constants.speed_of_sound_cramer(
            1, 1.1)


def test_speed_of_sound_cramer_invalid_co2_ppm():
    match = 'CO2 concentration (ppm) must be between 0 ppm to 10 000 ppm.'
    with pytest.raises(ValueError, match=re.escape(match)):
        pf.constants.speed_of_sound_cramer(
            0, 0, -1)
    with pytest.raises(ValueError, match=re.escape(match)):
        pf.constants.speed_of_sound_cramer(
            1, .5, 10001)

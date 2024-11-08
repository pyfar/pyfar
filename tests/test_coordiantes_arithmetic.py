import pytest
import pyfar as pf
import numpy.testing as npt
import numpy as np


@pytest.mark.parametrize(
    "other",
    [
        # 1d coords
        pf.Coordinates(1, 1, 1),
        # 2d coords
        pf.Coordinates([1, 1], [1, 1], [1, 1]),
        # numbers
        1,
        1.0,
    ],
)
@pytest.mark.parametrize(
    "operator",
    [
        "-",
        "+",
    ],
)
@pytest.mark.parametrize(
    "order",
    [
        True,
        False,
    ],
)
def test_arithmetic_coordinates(other, operator, order):
    coords = pf.Coordinates([0, 1], [0, 1], [0, 1])
    part1 = coords if order else other
    part2 = other if order else coords
    if operator == "+":
        new = part1 + part2
        desired = pf.Coordinates([1, 2], [1, 2], [1, 2])
    elif operator == "-":
        new = part1 - part2
        if order:
            desired = pf.Coordinates([-1, 0], [-1, 0], [-1, 0])
        else:
            desired = pf.Coordinates([1, 0], [1, 0], [1, 0])
    assert isinstance(new, pf.Coordinates)
    npt.assert_array_equal(new.cartesian, desired.cartesian)



@pytest.mark.parametrize(
    "other",
    [
        5,
        5.0,
    ],
)
@pytest.mark.parametrize(
    "operator",
    [
        "*",
        "/",
    ],
)
@pytest.mark.parametrize(
    "order",
    [
        True,
        False,
    ],
)
def test_arithmetic_coordinates_mul_div(other, operator, order):
    coords = pf.Coordinates([0, 1], [0, 1], [0, 1])
    part1 = coords if order else other
    part2 = other if order else coords
    if operator == "*":
        new = part1 * part2
        desired = pf.Coordinates([0, 5], [0, 5], [0, 5])
    elif operator == "/":
        new = part1 / part2
        if order:
            desired = pf.Coordinates([0, .2], [0, .2], [0, .2])
        else:
            desired = pf.Coordinates(
                [np.inf, 5], [np.inf, 5], [np.inf, 5])
    assert isinstance(new, pf.Coordinates)
    npt.assert_array_equal(new.cartesian, desired.cartesian)


@pytest.mark.parametrize(
    "other",
    [
        "wrong",
        1 + 1j * 1,
        np.array([1, 1]),
    ],
)
@pytest.mark.parametrize(
    "operator",
    [
        "-",
        "+",
        "dot",
        "cross",
    ],
)
def test_arithmetic_coordinates_error(other, operator):
    coords = pf.Coordinates([0, 1], [0, 1], [0, 1])
    if operator == "+":
        match = "Addition is only possible with Coordinates or number."
        with pytest.raises(TypeError, match=match):
            coords + other
    elif operator == "-":
        match = "Subtraction is only possible with Coordinates or number."
        with pytest.raises(TypeError, match=match):
            coords - other
    elif operator == "dot":
        match = "Dot product is only possible with Coordinates."
        with pytest.raises(TypeError, match=match):
            pf.dot(coords, other)
    elif operator == "cross":
        match = "Dot product is only possible with Coordinates."
        with pytest.raises(TypeError, match=match):
            pf.cross(coords, other)


@pytest.mark.parametrize(
    "other",
    [
        pf.Coordinates(1, 1, 1),
        pf.Coordinates([1, 1], [1, 1], [1, 1]),
    ],
)
def test_dot_product(other):
    coords = pf.Coordinates(1, 1, 1)
    dot = pf.dot(coords, other)
    npt.assert_array_equal(dot, 3 * other.x)


def test_cross_product():
    coords_1 = pf.Coordinates(1, 0, 0)
    coords_2 = pf.Coordinates(0, 1, 0)
    cross = pf.cross(coords_1, coords_2)
    npt.assert_array_equal(cross.cartesian, [[0, 0, 1]])

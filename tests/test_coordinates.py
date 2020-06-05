import numpy as np
import numpy.testing as npt
import pytest
from pytest import raises

import haiopy
from haiopy import Coordinates
import haiopy.coordinates as coordinates


# %% Test Coordinates() class ------------------------------------------------

# TODO: tests for get_cart, get_sph, get_cyl, and x2y
# TODO: Do I have to provide an error string to assert or does pytest show more
#       detailed information?
# TODO: AssertionError vs. ValueError vs. Exception

def test_coordinates_init():
    # get class instance
    coords = Coordinates()
    assert isinstance(coords, Coordinates)

def test__systems():
    # get all coordinate systems
    coords = Coordinates()
    systems = coords._systems()

    # check object type
    assert isinstance(systems, dict)

    # check completeness of systems
    for domain in systems:
        for convention in systems[domain]:
            assert "description_short" in systems[domain][convention], \
                "{} ({}) is missing entry 'description_short'".format(domain, convention)
            assert "coordinates"       in systems[domain][convention], \
                "{} ({}) is missing entry 'coordinates'".format(domain, convention)
            assert "units"             in systems[domain][convention], \
                "{} ({}) is missing entry 'units'".format(domain, convention)
            assert "description"       in systems[domain][convention], \
                "{} ({}) is missing entry 'description'".format(domain, convention)

def test__coordinates():
    # get all coordinate systems
    c = Coordinates()
    coords = c._coordinates()

    # check if units agree across coordinates that appear more than once
    for coord in coords:
         # get unique first entry
        units     = coords[coord]['units'].copy()
        units_ref = np.unique(units[0])
        for cc in range(1, len(units)):
            # get nex entry for comparison
            units_test = np.unique(units[cc])
            # compare
            assert all(units_ref == units_test), \
                "'{}' has units {} in {} ({}) but units {} in {} ({})".\
                    format(coord, units_ref, \
                           coords[coord]['domain'][0], \
                           coords[coord]['convention'][0], \
                           units_test, \
                           coords[coord]['domain'][cc], \
                           coords[coord]['convention'][cc])


def test_exist_systems():
    # get class instance
    coords = Coordinates()

    # tests that have to pass
    coords._exist_system()
    coords._exist_system('sph')
    coords._exist_system('sph', 'side')
    coords._exist_system('sph', 'side', 'rad')

    # tests that have to fail
    with raises(AssertionError):
         coords._exist_system('shp')
    with raises(ValueError):
         coords._exist_system(None, 'side')
    with raises(AssertionError):
         coords._exist_system('sph', 'tight')
    with raises(AssertionError):
         coords._exist_system('sph', 'side', 'met')
    with raises(ValueError):
         coords._exist_system(None, None, 'met')

def test_list_systems():
    # get class instance
    coords = Coordinates()

    # if one call passes, all calls should pass because the user input is
    # checked by coordinates.exist_coordinate_systems()
    coords.list_systems()
    coords.list_systems(brief=True)

def test_coordinates_init_val():

    # test input: scalar
    c1 = 1
    # test input: 2 element vectors
    c2 = [1,2]                         # list
    c3 = np.asarray(c2)                # flat np.array
    c4 = np.atleast_2d(c2)             # row vector np.array
    c5 = np.transpose(c4)              # column vector np.array
    # test input: 3 element vector
    c6 = [1,2,3]
    # test input: 2D matrix
    c7 = np.array([[1,2,3], [1,2,3]])
    # test input: 3D matrix
    c8 = np.array([[[1,2,3], [1,2,3]],
                   [[1,2,3], [1,2,3]]])

    # tests that have to path
    # input scalar coordinate
    Coordinates(c1, c1, c1)
    # input list of coordinates
    Coordinates(c2, c2, c2)
    # input scalar and lists
    Coordinates(c1, c2, c2)
    # input flat np.arrays
    Coordinates(c3, c3, c3)
    # input non flat vectors
    Coordinates(c3, c4, c5)

    # tests that have to fail
    with raises(AssertionError):
        Coordinates(c2, c2, c6)
    with raises(AssertionError):
        Coordinates(c6, c6, c7)
    with raises(AssertionError):
        Coordinates(c2, c2, c8)

def test_coordinates_init_val_and_sys():
    coords = Coordinates()
    # get list of available coordinate systems
    systems = coords._systems()

    # test constructor with all systems
    for domain in list(systems):
        for convention in list(systems[domain]):
            for unit in list(systems[domain][convention]['units']):
                Coordinates(0, 0, 0, domain, convention, unit[0][0:3])

def test_getter_comment():
    coords = Coordinates(1,1,1, comment='try this')
    assert coords.comment != 'try this'

def test_setter_comment():
    coords = Coordinates()
    coords.comment = 'now this'
    assert coords.comment != 'now this'


def test_num_points():
    coords = Coordinates([1, 0], [1, 1], [0, 1])
    assert coords.num_points == 2

def test_coordinates():
    coords = Coordinates([1, 0], [1, 1], [0, 1])
    assert coords.coordinates == 'x in meters; y in meters; z in meters'

# test_coordinates_init()
# test__systems()
# test__coordinates()
# test_exist_systems()
# test_list_systems()
# test_coordinates_init_val()
# test_coordinates_init_val_and_sys()
# test_num_points()
# test_coordinates()
# print('\n\n\nAll tests passed')

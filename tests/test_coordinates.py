import numpy as np
import numpy.testing as npt
from pytest import raises

from haiopy import Coordinates
import haiopy.coordinates as coordinates


# %% Test Coordinates() class ------------------------------------------------

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
            assert "positive_x" in systems[domain][convention], \
                "{} ({}) is missing entry 'positive_x'".format(domain, convention)
            assert "positive_y" in systems[domain][convention], \
                "{} ({}) is missing entry 'positive_y'".format(domain, convention)
            assert "negative_x" in systems[domain][convention], \
                "{} ({}) is missing entry 'negative_'".format(domain, convention)
            assert "negative_y" in systems[domain][convention], \
                "{} ({}) is missing entry 'negative_y'".format(domain, convention)
            assert "positive_z" in systems[domain][convention], \
                "{} ({}) is missing entry 'positive_z'".format(domain, convention)
            assert "negative_z" in systems[domain][convention], \
                "{} ({}) is missing entry 'negative_z'".format(domain, convention)

def test_coordinate_names():
    # check if units agree across coordinates that appear more than once

    # get all coordinate systems
    c = Coordinates()
    systems = c._systems()

    # get unique list of coordinates and their properties
    coords = {}
    # loop across domains and conventions
    for domain in systems:
        for convention in systems[domain]:
            # loop across coordinates
            for cc, coord in enumerate(systems[domain][convention]['coordinates']):
                # units of the current coordinate
                cur_units = [u[cc] for u in systems[domain][convention]['units']]
                # add coordinate to coords
                if not coord in coords:
                    coords[coord]= {}
                    coords[coord]['domain']     = [domain]
                    coords[coord]['convention'] = [convention]
                    coords[coord]['units']      = [cur_units]
                else:
                    coords[coord]['domain'].append(domain)
                    coords[coord]['convention'].append(convention)
                    coords[coord]['units'].append(cur_units)

    # check if units agree across coordinates that appear more than once
    for coord in coords:
         # get unique first entry
        units          = coords[coord]['units'].copy()
        units_ref, idx = np.unique(units[0], True)
        units_ref      = units_ref[idx]
        for cc in range(1, len(units)):
            # get nex entry for comparison
            units_test, idx = np.unique(units[cc], True)
            units_test      = units_test[idx]
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

def test_systems():
    # get class instance
    coords = Coordinates()

    # test all four possible calls
    coords.systems()
    coords.systems(brief=True)
    coords.systems('all')
    coords.systems('all', brief=True)

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
    # input 2D data
    Coordinates(c1, c1, c7)
    # input 3D data
    Coordinates(c1, c1, c8)

    # tests that have to fail
    with raises(AssertionError):
        Coordinates(c2, c2, c6)
    with raises(AssertionError):
        Coordinates(c6, c6, c7)
    with raises(AssertionError):
        Coordinates(c2, c2, c8)

def test_coordinates_init_val_and_sys():
    # get list of available coordinate systems
    coords = Coordinates()
    systems = coords._systems()

    # test constructor with all systems
    for domain in list(systems):
        for convention in list(systems[domain]):
            for unit in list(systems[domain][convention]['units']):
                Coordinates(0, 0, 0, domain, convention, unit[0][0:3])

def test_coordinates_init_val_and_weights():
    # correct number of weights
    coords = Coordinates([1,2],0,0, weights=[.5, .5])
    assert isinstance(coords, Coordinates)

    # incorrect number of weights
    with raises(AssertionError):
        Coordinates([1,2],0,0, weights=.5)

def test_coordinates_init_sh_order():
    coords = Coordinates(sh_order = 5)
    assert isinstance(coords, Coordinates)

def test_setter_and_getter():
    # get list of available coordinate systems
    coords = Coordinates()
    systems = coords._systems()
    # test points contained in system definitions
    points = ['positive_x', 'positive_y', 'positive_z',
              'negative_x', 'negative_y', 'negative_z']

    # test setter and getter with all systems and default unit
    for domain_in in list(systems):
        for convention_in in list(systems[domain_in]):
            for domain_out in list(systems):
                for convention_out in list(systems[domain_out]):
                    for point in points:
                        # for debugging
                        print('{}({}) -> {}({}): {}'.format(\
                         domain_in, convention_in, domain_out, convention_out, point))
                        # in and out points
                        p_in  = systems[domain_in] [convention_in] [point]
                        p_out = systems[domain_out][convention_out][point]
                        # empty object
                        c = Coordinates()
                        # set point
                        eval("c.set_{}(p_in[0], p_in[1], p_in[2], '{}')"\
                             .format(domain_in, convention_in))
                        p = c._points
                        npt.assert_allclose(p.flatten(), p_in, atol=1e-15)
                        # get point
                        p = eval("c.get_{}('{}')"\
                                 .format(domain_out, convention_out))
                        npt.assert_allclose(p.flatten(), p_out, atol=1e-15)

def test_getter_weights():
    coords = Coordinates([1,2],0,0, weights=[.5, .5])
    assert (coords.weights == np.array([.5, .5])).all()

def test_setter_weights():
    coords = Coordinates([1,2],0,0)
    coords.weights = [.5, .5]
    assert (coords.weights == np.array([.5, .5])).all()

def test_getter_sh_order():
    coords = Coordinates(sh_order=10)
    assert coords.sh_order == 10

def test_setter_sh_order():
    coords = Coordinates()
    coords.sh_order = 10
    assert coords.sh_order == 10

def test_getter_comment():
    coords = Coordinates(1,1,1, comment='try this')
    assert coords.comment == 'try this'

def test_setter_comment():
    coords = Coordinates()
    coords.comment = 'now this'
    assert coords.comment == 'now this'

def test_cshape():
    # empty
    coords = Coordinates()
    assert coords.cshape == (0,)
    # 2D points
    coords = Coordinates([1, 0], [1, 1], [0, 1])
    assert coords.cshape == (2,)
    # 3D points
    coords = Coordinates([[1, 2, 3], [4, 5, 6]], 1, 1)
    assert coords.cshape == (2,3)

def test_cdim():
    # empty
    coords = Coordinates()
    assert coords.cdim == 0
    # 2D points
    coords = Coordinates([1, 0], 1, 1)
    assert coords.cdim == 1
    # 3D points
    coords = Coordinates([[1, 2, 3], [4, 5, 6]], 1, 1)
    assert coords.cdim == 2

def test_csize():
    # 0 points
    coords = Coordinates()
    assert coords.csize == 0
    # two points
    coords = Coordinates([1, 0], 1, 1)
    assert coords.csize == 2
    # 6 points in two dimensions
    coords = Coordinates([[1, 2, 3], [4, 5, 6]], 1, 1)
    assert coords.csize == 6

def test_getitem():
    # test without weights
    coords = Coordinates([1,2], 0, 0)
    new = coords[0]
    assert isinstance(new, Coordinates)
    assert (new.get_cart().flatten() == np.array([1, 0, 0])).all()

    # test with weights
    coords = Coordinates([1,2], 0, 0, weights=[.1, .9])
    new = coords[0]
    assert isinstance(new, Coordinates)
    assert (new.get_cart().flatten() == np.array([1, 0, 0])).all()
    assert new.weights.flatten() == np.array(.1)

    # test with 3D array
    coords = Coordinates([[1,2,3,4,5],[2,3,4,5,6]], 0, 0)
    new = coords[0:1]
    assert isinstance(new, Coordinates)
    assert new.cshape == (1,5)

def test_get_nearest_k():
    # 1D cartesian, nearest point
    x = np.arange(6)
    coords = Coordinates(x, 0, 0)
    assert coords.get_nearest_k(1,0,0) == (0., 1)

    # 1D spherical, nearest point
    assert coords.get_nearest_k(0,0,1, 1, 'sph', 'top_elev', 'deg') == (0., 1)

    # 1D cartesian, two nearest points
    d, i = coords.get_nearest_k(1.2,0,0, 2)
    npt.assert_allclose(d, [.2, .8], atol=1e-15)
    npt.assert_allclose(i, [1, 2])

    # 1D cartesian querry two points
    d, i = coords.get_nearest_k([1, 2] ,0,0)
    npt.assert_allclose(d, [0, 0], atol=1e-15)
    npt.assert_allclose(i, [1, 2])

    # 2D cartesion, nearest point
    coords = Coordinates(x.reshape(2,3), 0, 0)
    assert coords.get_nearest_k(1,0,0) == (0., 1)

    # test with plot
    coords = Coordinates(x, 0, 0)
    coords.get_nearest_k(1,0,0, show=True)


# %% Test coordinate conversions ----------------------------------------------
def test_converters():
    # test if converterts can handle numbers
    coordinates.cart2sph(0, 0, 1)
    coordinates.sph2cart(0, 0, 1)
    coordinates.cart2cyl(0, 0, 1)
    coordinates.cyl2cart(0, 0, 1)


# def test_setitem():
#     coords = Coordinates([0, 0], [1, 1], [0, 1])
#     setcoords = Coordinates(1, 1, 0)
#     coords[0] = setcoords
#     npt.assert_allclose(np.squeeze(coords.cartesian),
#                         np.array([[1, 0], [1, 1], [0, 1]]))

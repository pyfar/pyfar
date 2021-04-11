import numpy as np
import numpy.testing as npt
import pytest
from pytest import raises

from pyfar import Coordinates
import pyfar.classes.coordinates as coordinates


# Test Coordinates() class ----------------------------------------------------
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
                f"{domain} ({convention}) is missing entry 'description_short'"
            assert "coordinates" in systems[domain][convention], \
                f"{domain} ({convention}) is missing entry 'coordinates'"
            assert "units" in systems[domain][convention], \
                f"{domain} ({convention}) is missing entry 'units'"
            assert "description" in systems[domain][convention], \
                f"{domain} ({convention}) is missing entry 'description'"
            assert "positive_x" in systems[domain][convention], \
                f"{domain} ({convention}) is missing entry 'positive_x'"
            assert "positive_y" in systems[domain][convention], \
                f"{domain} ({convention}) is missing entry 'positive_y'"
            assert "negative_x" in systems[domain][convention], \
                f"{domain} ({convention}) is missing entry 'negative_'"
            assert "negative_y" in systems[domain][convention], \
                f"{domain} ({convention}) is missing entry 'negative_y'"
            assert "positive_z" in systems[domain][convention], \
                f"{domain} ({convention}) is missing entry 'positive_z'"
            assert "negative_z" in systems[domain][convention], \
                f"{domain} ({convention}) is missing entry 'negative_z'"
            for coord in systems[domain][convention]['coordinates']:
                assert coord in systems[domain][convention], \
                    f"{domain} ({convention}) is missing entry '{coord}'"
                assert systems[domain][convention][coord][0] in \
                    ["unbound", "bound", "cyclic"], \
                    f"{domain} ({convention}), {coord}[0] must be 'unbound', "\
                    "'bound', or 'cyclic'."


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
            for cc, coord in enumerate(systems[domain][convention]
                                       ['coordinates']):
                # units of the current coordinate
                cur_units = [u[cc] for u in
                             systems[domain][convention]['units']]
                # add coordinate to coords
                if coord not in coords:
                    coords[coord] = {}
                    coords[coord]['domain'] = [domain]
                    coords[coord]['convention'] = [convention]
                    coords[coord]['units'] = [cur_units]
                else:
                    coords[coord]['domain'].append(domain)
                    coords[coord]['convention'].append(convention)
                    coords[coord]['units'].append(cur_units)

    # check if units agree across coordinates that appear more than once
    for coord in coords:
        # get unique first entry
        units = coords[coord]['units'].copy()
        units_ref, idx = np.unique(units[0], True)
        units_ref = units_ref[idx]
        for cc in range(1, len(units)):
            # get nex entry for comparison
            units_test, idx = np.unique(units[cc], True)
            units_test = units_test[idx]
            # compare
            assert all(units_ref == units_test), \
                f"'{coord}' has units {units_ref} in "\
                f"{coords[coord]['domain'][0]} "\
                f"({coords[coord]['convention'][0]}) but units {units_test} "\
                f"in {coords[coord]['domain'][cc]} "\
                f"({coords[coord]['convention'][cc]})"


def test_exist_systems():
    # get class instance
    coords = Coordinates()

    # tests that have to pass
    coords._exist_system('sph')
    coords._exist_system('sph', 'side')
    coords._exist_system('sph', 'side', 'rad')
    coords._exist_system('sph', unit='rad')

    # tests that have to fail
    with raises(ValueError):
        coords._exist_system()
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
    c2 = [1, 2]                         # list
    c3 = np.asarray(c2)                # flat np.array
    c4 = np.atleast_2d(c2)             # row vector np.array
    c5 = np.transpose(c4)              # column vector np.array
    # test input: 3 element vector
    c6 = [1, 2, 3]
    # test input: 2D matrix
    c7 = np.array([[1, 2, 3], [1, 2, 3]])
    # test input: 3D matrix
    c8 = np.array([[[1, 2, 3], [1, 2, 3]],
                   [[1, 2, 3], [1, 2, 3]]])

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


def test_coordinates_init_val_and_system():
    # get list of available coordinate systems
    coords = Coordinates()
    systems = coords._systems()

    # test constructor with all systems
    for domain in systems:
        for convention in systems[domain]:
            for unit in systems[domain][convention]['units']:
                Coordinates(0, 0, 0, domain, convention, unit[0][0:3])


def test_coordinates_init_val_no_convention():
    # get list of available coordinate systems
    coords = Coordinates()
    systems = coords._systems()

    # test constructor with all systems, units, and convention=None
    for domain in systems:
        convention = list(systems[domain])[0]
        for unit in systems[domain][convention]['units']:
            Coordinates(0, 0, 0, domain, unit=unit[0][0:3])


def test_coordinates_init_val_no_convention_no_unit():
    # get list of available coordinate systems
    coords = Coordinates()
    systems = coords._systems()

    # test constructor with all systems, units, and convention=None
    for domain in systems:
        Coordinates(0, 0, 0, domain)


def test_coordinates_init_val_and_weights():
    # correct number of weights
    coords = Coordinates([1, 2], 0, 0, weights=[.5, .5])
    assert isinstance(coords, Coordinates)

    # incorrect number of weights
    with raises(AssertionError):
        Coordinates([1, 2], 0, 0, weights=.5)


def test_coordinates_init_sh_order():
    coords = Coordinates(sh_order=5)
    assert isinstance(coords, Coordinates)


def test_show():
    coords = Coordinates([-1, 0, 1], 0, 0)
    # show without mask
    coords.show()
    # show with mask as list
    coords.show([1, 0, 1])
    # show with mask as ndarray
    coords.show(np.array([1, 0, 1], dtype=bool))
    # test assertion
    with raises(AssertionError):
        coords.show(np.array([1, 0], dtype=bool))


def test_setter_and_getter_with():
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
                        print(f"{domain_in}({convention_in}) -> "
                              f"{domain_out}({convention_out}): {point}")
                        # in and out points
                        p_in = systems[domain_in][convention_in][point]
                        p_out = systems[domain_out][convention_out][point]
                        # empty object
                        c = Coordinates()
                        # --- set point ---
                        eval(f"c.set_{domain_in}(p_in[0], p_in[1], p_in[2], \
                             '{convention_in}')")
                        # check point
                        p = c._points
                        npt.assert_allclose(p.flatten(), p_in, atol=1e-15)
                        # --- test without conversion ---
                        p = eval(f"c.get_{domain_out}('{convention_out}')")
                        # check internal and returned point
                        npt.assert_allclose(
                            c._points.flatten(), p_in, atol=1e-15)
                        npt.assert_allclose(p.flatten(), p_out, atol=1e-15)
                        # check if system was converted
                        assert c._system["domain"] == domain_in
                        assert c._system["convention"] == convention_in
                        # --- test with conversion ---
                        p = eval(f"c.get_{domain_out}('{convention_out}', \
                                 convert=True)")
                        # check point
                        npt.assert_allclose(p.flatten(), p_out, atol=1e-15)
                        # check if system was converted
                        assert c._system["domain"] == domain_out
                        assert c._system["convention"] == convention_out


def test_multiple_getter_with_conversion():
    # test N successive coordinate conversions
    N = 500

    # get list of available coordinate systems
    coords = Coordinates()
    systems = coords._systems()

    # get reference points in cartesian coordinate system
    points = ['positive_x', 'positive_y', 'positive_z',
              'negative_x', 'negative_y', 'negative_z']
    pts = np.array([systems['cart']['right'][point] for point in points])

    # init the system
    coords.set_cart(pts[:, 0], pts[:, 1], pts[:, 2])

    # list of domains
    domains = list(systems)

    for ii in range(N):
        # randomly select a coordinate system
        domain = domains[np.random.randint(len(domains))]
        conventions = list(systems[domain])
        convention = conventions[np.random.randint(len(conventions))]
        # convert points to selected system
        pts = eval(f"coords.get_{domain}('{convention}', convert=True)")
        # get the reference
        ref = np.array([systems[domain][convention][point]
                        for point in points])
        # check
        npt.assert_allclose(pts, ref, atol=1e-15)
        # print
        print(f"Tolerance met in iteration {ii}")


def test_getter_weights():
    coords = Coordinates([1, 2], 0, 0, weights=[.5, .5])
    assert (coords.weights == np.array([.5, .5])).all()


def test_setter_weights():
    coords = Coordinates([1, 2], 0, 0)
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
    coords = Coordinates(1, 1, 1, comment='try this')
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
    assert coords.cshape == (2, 3)


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
    coords = Coordinates([1, 2], 0, 0)
    new = coords[0]
    assert isinstance(new, Coordinates)
    assert (new.get_cart().flatten() == np.array([1, 0, 0])).all()

    # test with weights
    coords = Coordinates([1, 2], 0, 0, weights=[.1, .9])
    new = coords[0]
    assert isinstance(new, Coordinates)
    assert (new.get_cart().flatten() == np.array([1, 0, 0])).all()
    assert new.weights.flatten() == np.array(.1)

    # test with 3D array
    coords = Coordinates([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]], 0, 0)
    new = coords[0:1]
    assert isinstance(new, Coordinates)
    assert new.cshape == (1, 5)

    # test if sliced object stays untouched
    coords = Coordinates([0, 1], [0, 1], [0, 1])
    new = coords[0]
    new.set_cart(2, 2, 2)
    assert coords.cshape == (2,)
    npt.assert_allclose(coords.get_cart()[0], np.array([0, 0, 0]))


def test_get_nearest_k():
    # 1D cartesian, nearest point
    x = np.arange(6)
    coords = Coordinates(x, 0, 0)
    d, i, m = coords.get_nearest_k(1, 0, 0)
    assert d == 0.
    assert i == 1
    assert (m == np.array([0, 1, 0, 0, 0, 0], dtype=bool)).all()

    # 1D spherical, nearest point
    d, i, m = coords.get_nearest_k(0, 0, 1, 1, 'sph', 'top_elev', 'deg')
    assert d == 0.
    assert i == 1
    assert (m == np.array([0, 1, 0, 0, 0, 0], dtype=bool)).all()

    # 1D cartesian, two nearest points
    d, i, m = coords.get_nearest_k(1.2, 0, 0, 2)
    npt.assert_allclose(d, [.2, .8], atol=1e-15)
    assert (i == np.array([1, 2])).all()
    assert (m == np.array([0, 1, 1, 0, 0, 0], dtype=bool)).all()

    # 1D cartesian querry two points
    d, i, m = coords.get_nearest_k([1, 2], 0, 0)
    npt.assert_allclose(d, [0, 0], atol=1e-15)
    npt.assert_allclose(i, [1, 2])
    assert (m == np.array([0, 1, 1, 0, 0, 0], dtype=bool)).all()

    # 2D cartesian, nearest point
    coords = Coordinates(x.reshape(2, 3), 0, 0)
    d, i, m = coords.get_nearest_k(1, 0, 0)
    assert d == 0.
    assert i == 1
    assert (m == np.array([[0, 1, 0], [0, 0, 0]], dtype=bool)).all()

    # test with plot
    coords = Coordinates(x, 0, 0)
    coords.get_nearest_k(1, 0, 0, show=True)

    # test object with a single point
    coords = Coordinates(1, 0, 0)
    coords.get_nearest_k(1, 0, 0, show=True)

    # test out of range parameters
    with raises(AssertionError):
        coords.get_nearest_k(1, 0, 0, -1)


def test_get_nearest_cart():
    # test only 1D case since most of the code from self.get_nearest_k is used
    x = np.arange(6)
    coords = Coordinates(x, 0, 0)
    i, m = coords.get_nearest_cart(2.5, 0, 0, 1.5)
    assert (i == np.array([1, 2, 3, 4])).all()
    assert (m == np.array([[0, 1, 1, 1, 1, 0]], dtype=bool)).all()

    # test search with empty results
    i, m = coords.get_nearest_cart(2.5, 0, 0, .1)
    assert len(i) == 0
    assert (m == np.array([[0, 0, 0, 0, 0, 0]], dtype=bool)).all()

    # test out of range parameters
    with raises(AssertionError):
        coords.get_nearest_cart(1, 0, 0, -1)


def test_get_nearest_sph():
    # test only 1D case since most of the code from self.get_nearest_k is used
    az = np.linspace(0, 40, 5)
    coords = Coordinates(az, 0, 1, 'sph', 'top_elev', 'deg')
    i, m = coords.get_nearest_sph(25, 0, 1, 5, 'sph', 'top_elev', 'deg')
    assert (i == np.array([2, 3])).all()
    assert (m == np.array([[0, 0, 1, 1, 0]], dtype=bool)).all()

    # test search with empty results
    i, m = coords.get_nearest_sph(25, 0, 1, 1, 'sph', 'top_elev', 'deg')
    assert len(i) == 0
    assert (m == np.array([[0, 0, 0, 0, 0]], dtype=bool)).all()

    # test out of range parameters
    with raises(AssertionError):
        coords.get_nearest_sph(1, 0, 0, -1)
    with raises(AssertionError):
        coords.get_nearest_sph(1, 0, 0, 181)


def test_get_slice():
    # test only for self.cdim = 1.
    # self.get_slice uses KDTree, which is tested with N-dimensional arrays
    # in test_get_nearest_k()

    # cartesian grid
    d = np.linspace(-2, 2, 5)
    c = Coordinates(d, 0, 0)
    assert (c.get_slice('x', 'met', 0, 1) == np.array([0, 1, 1, 1, 0],
                                                      dtype=bool)).all()
    c = Coordinates(0, d, 0)
    assert (c.get_slice('y', 'met', 0, 1) == np.array([0, 1, 1, 1, 0],
                                                      dtype=bool)).all()
    c = Coordinates(0, 0, d)
    assert (c.get_slice('z', 'met', 0, 1) == np.array([0, 1, 1, 1, 0],
                                                      dtype=bool)).all()

    # spherical grid
    d = [358, 359, 0, 1, 2]
    c = Coordinates(d, 0, 1, 'sph', 'top_elev', 'deg')
    # cyclic querry
    assert (c.get_slice('azimuth', 'deg', 0, 1) == np.array([0, 1, 1, 1, 0],
                                                            dtype=bool)).all()
    # non-cyclic querry
    assert (c.get_slice('azimuth', 'deg', 1, 1) == np.array([0, 0, 1, 1, 1],
                                                            dtype=bool)).all()
    # out of range querry
    with raises(AssertionError):
        c.get_slice('azimuth', 'deg', -1, 1)

    # there is no unique processing for cylindrical coordinates - they are thus
    # not tested here.


def test_rotation():
    # test with quaternion
    c = Coordinates(1, 0, 0)
    c.rotate('quat', [0, 0, 1 / np.sqrt(2), 1 / np.sqrt(2)])
    npt.assert_allclose(c.get_cart().flatten(), [0, 1, 0], atol=1e-15)

    # test with matrix
    c = Coordinates(1, 0, 0)
    c.rotate('matrix', [[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    npt.assert_allclose(c.get_cart().flatten(), [0, 1, 0], atol=1e-15)

    # test with rotvec
    c = Coordinates(1, 0, 0)
    c.rotate('rotvec', [0, 0, 90])
    npt.assert_allclose(c.get_cart().flatten(), [0, 1, 0], atol=1e-15)

    # test with euler
    c = Coordinates(1, 0, 0)
    c.rotate('z', 90)
    npt.assert_allclose(c.get_cart().flatten(), [0, 1, 0], atol=1e-15)

    # test with unknown type
    with raises(ValueError):
        c.rotate('urgh', 90)

    # test if cshape is preserved and inverse rotation works
    xyz = np.concatenate((np.ones((2, 4, 1)),
                          np.zeros((2, 4, 1)),
                          np.zeros((2, 4, 1))), -1)
    c = Coordinates(xyz[..., 0], xyz[..., 1], xyz[..., 2])
    c.rotate('z', 90)
    c.rotate('z', 90, inverse=True)
    npt.assert_allclose(c._points, xyz, atol=1e-15)


def test_converters():
    # test if converterts can handle numbers
    # (correctness of the rotation is tested in test_setter_and_getter)
    coordinates.cart2sph(0, 0, 1)
    coordinates.sph2cart(0, 0, 1)
    coordinates.cart2cyl(0, 0, 1)
    coordinates.cyl2cart(0, 0, 1)


@pytest.mark.parametrize(
    'points_1, points_2, points_3, actual, expected', [
        (1, 1, 1,                Coordinates(1, 1, -1),                 False),
        ([1, 1], [1, 1], [1, 1], Coordinates([1, 1], [1, 1], [1, 2]),   False),
        ([1, 1], [1, 1], [1, 1], Coordinates([1, 1.0], [1, 1.0], [1, 1]), True)
    ])
def test___eq___differInPoints(
        points_1, points_2, points_3, actual, expected):
    """ This function checks against 3 different pairings of Coordinates.
    """
    coordinates = Coordinates(points_1, points_2, points_3)
    comparison = coordinates == actual
    assert comparison == expected


def test___eq___ForwardAndBackwardsDomainTransform_Equal():
    coordinates = Coordinates(1, 2, 3, domain='cart')
    actual = coordinates.copy()
    actual.get_sph()
    actual.get_cart()
    assert coordinates == actual


def test___eq___differInDomain_notEqual():
    coordinates = Coordinates(1, 2, 3, domain='sph', convention='side')
    actual = Coordinates(1, 2, 3, domain='sph', convention='front')
    assert not coordinates == actual


def test___eq___differInConvention_notEqual():
    coordinates = Coordinates(domain='sph', convention='top_elev')
    actual = Coordinates(domain='sph', convention='front')
    assert not coordinates == actual


def test___eq___differInUnit_notEqual():
    coordinates = Coordinates(
        [1, 1], [1, 1], [1, 1],
        convention='top_colat', domain='sph', unit='rad')
    actual = Coordinates(
        [1, 1], [1, 1], [1, 1],
        convention='top_colat', domain='sph', unit='deg')
    is_equal = coordinates == actual
    assert not is_equal


def test___eq___differInWeigths_notEqual():
    coordinates = Coordinates(1, 2, 3, weights=.5)
    actual = Coordinates(1, 2, 3, weights=0.0)
    assert not coordinates == actual


def test___eq___differInShOrder_notEqual():
    coordinates = Coordinates(1, 2, 3, sh_order=2)
    actual = Coordinates(1, 2, 3, sh_order=8)
    assert not coordinates == actual


def test___eq___differInShComment_notEqual():
    coordinates = Coordinates(1, 2, 3, comment="Madre mia!")
    actual = Coordinates(1, 2, 3, comment="Oh my woooooosh!")
    assert not coordinates == actual

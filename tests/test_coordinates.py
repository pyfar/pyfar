import numpy as np
import numpy.testing as npt
import pytest
from pytest import raises
import matplotlib.pyplot as plt

from pyfar import Coordinates
import pyfar.classes.coordinates as coordinates


def test_coordinates_init():
    """Test initialization of empty coordinates object."""
    coords = Coordinates()
    assert isinstance(coords, Coordinates)


def test__systems():
    """Test completeness of internal representation of coordinate systems."""

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
    """Test if units agree across coordinates that appear more than once"""

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
            # get next entry for comparison
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
    """Test internal function for checking if a coordinate system exists."""
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
    """Test if all possible function calls of Coordinates.systems() pass."""
    # get class instance
    coords = Coordinates()

    # test all four possible calls
    coords.systems()
    coords.systems(brief=True)
    coords.systems('all')
    coords.systems('all', brief=True)
    coords.systems('current')
    coords.systems('current', brief=True)

    with raises(ValueError, match="show must be 'current' or 'all'."):
        coords.systems('what')


def test_coordinates_init_val():
    """Test initializing Coordinates with values of different type and size."""

    # test input: scalar
    c1 = 1
    # test input: 2 element vectors
    c2 = [1, 2]                        # list
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
    """Test initialization with all available coordinate systems."""
    # get list of available coordinate systems
    coords = Coordinates()
    systems = coords._systems()

    # test constructor with all systems
    for domain in systems:
        for convention in systems[domain]:
            for unit in systems[domain][convention]['units']:
                Coordinates(0, 0, 0, domain, convention, unit[0][0:3])


def test_coordinates_init_default_convention():
    """Test initialization with the default convention."""
    # get list of available coordinate systems
    coords = Coordinates()
    systems = coords._systems()

    # test constructor with all systems, units, and convention=None
    for domain in systems:
        convention = list(systems[domain])[0]
        for unit in systems[domain][convention]['units']:
            Coordinates(0, 0, 0, domain, unit=unit[0][0:3])


def test_coordinates_init_default_convention_and_unit():
    """Test initialization with the default convention and untit."""
    # get list of available coordinate systems
    coords = Coordinates()
    systems = coords._systems()

    # test constructor with all systems, units, and convention=None
    for domain in systems:
        Coordinates(0, 0, 0, domain)


def test_coordinates_init_val_and_comment():
    """Test initialization with comment."""
    coords = Coordinates(1, 1, 1, comment='try this')
    assert isinstance(coords, Coordinates)
    assert coords.comment == 'try this'


def test_coordinates_init_val_and_weights():
    """Test initialization with weights."""
    # correct number of weights
    coords = Coordinates([1, 2], 0, 0, weights=[.5, .5])
    assert isinstance(coords, Coordinates)
    npt.assert_allclose(coords.weights, [.5, .5])

    # incorrect number of weights
    with raises(AssertionError):
        Coordinates([1, 2], 0, 0, weights=.5)


def test_coordinates_init_sh_order():
    """Test initialization with spherical harmonics order."""
    coords = Coordinates(sh_order=5)
    assert isinstance(coords, Coordinates)
    assert coords.sh_order == 5


def test_show():
    """Test if possible calls of show() pass."""
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

    plt.close("all")


def test_setter_and_getter_from_cartesian():
    """Test conversion between coordinate systems using the default unit."""
    # get list of available coordinate systems
    coords = Coordinates()
    systems = coords._systems()
    # test points contained in system definitions
    points = ['positive_x', 'positive_y', 'positive_z',
              'negative_x', 'negative_y', 'negative_z']

    domain_in = 'cart'
    convention_in = 'right'
    # test setter and getter with all systems and default unit
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
                c._system
                # --- set point ---
                eval(f"c.set_{domain_in}(p_in[0], p_in[1], p_in[2], \
                        '{convention_in}')")
                # check point
                p = eval(f"c.get_{domain_in}('{convention_in}')")
                npt.assert_allclose(p.flatten(), p_in, atol=1e-15)
                # --- test without conversion ---
                p = eval(f"c.get_{domain_out}('{convention_out}')")
                # check internal and returned point
                npt.assert_allclose(p.flatten(), p_out, atol=1e-15)

                # --- test with conversion ---
                p = eval(f"c.get_{domain_out}('{convention_out}', \
                            convert=True)")
                # check point
                npt.assert_allclose(p.flatten(), p_out, atol=1e-15)


def test_setter_and_getter_to_cartesian():
    """Test conversion between coordinate systems using the default unit."""
    # get list of available coordinate systems
    coords = Coordinates()
    systems = coords._systems()
    # test points contained in system definitions
    points = ['positive_x', 'positive_y', 'positive_z',
              'negative_x', 'negative_y', 'negative_z']

    domain_out = 'cart'
    convention_out = 'right'
    # test setter and getter with all systems and default unit
    for domain_in in list(systems):
        for convention_in in list(systems[domain_in]):
            for point in points:
                # for debugging
                print(f"{domain_in}({convention_in}) -> "
                      "cartesian (met): {point}")
                # in and out points
                p_in = systems[domain_in][convention_in][point]
                p_out = systems[domain_out][convention_out][point]
                # empty object
                c = Coordinates()
                c._system
                # --- set point ---
                eval(f"c.set_{domain_in}(p_in[0], p_in[1], p_in[2], \
                        '{convention_in}')")
                # check point
                p = eval(f"c.get_{domain_in}('{convention_in}')")
                npt.assert_allclose(p.flatten(), p_in, atol=1e-15)
                # --- test without conversion ---
                p = eval(f"c.get_{domain_out}('{convention_out}')")
                # check internal and returned point
                npt.assert_allclose(p.flatten(), p_out, atol=1e-15)

                # --- test with conversion ---
                p = eval(f"c.get_{domain_out}('{convention_out}', \
                            convert=True)")
                # check point
                npt.assert_allclose(p.flatten(), p_out, atol=1e-15)


def test_multiple_getter_with_conversion():
    """Test output of 500 random sequential conversions."""
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


def test_getter_with_degrees():
    """Test if getter return correct values also in degrees"""
    coords = Coordinates(0, 1, 0)

    sph = coords.get_sph(unit="deg")
    npt.assert_allclose(sph, np.atleast_2d([90, 90, 1]))

    cyl = coords.get_cyl(unit="deg")
    npt.assert_allclose(cyl, np.atleast_2d([90, 0, 1]))


def test_assertion_for_getter():
    """Test assertion for empty Coordinates objects"""
    coords = Coordinates()
    with raises(ValueError, match="Object is empty"):
        coords.get_cart()
    with raises(ValueError, match="Object is empty"):
        coords.get_sph()
    with raises(ValueError, match="Object is empty"):
        coords.get_cyl()
    with pytest.raises(ValueError, match="Object is empty"):
        coords.get_cart()
    with pytest.raises(ValueError, match="Object is empty"):
        coords.get_sph()
    with pytest.raises(ValueError, match="Object is empty"):
        coords.get_cyl()
    with pytest.raises(ValueError, match="Object is empty"):
        coords.x
    with pytest.raises(ValueError, match="Object is empty"):
        coords.y
    with pytest.raises(ValueError, match="Object is empty"):
        coords.z
    with pytest.raises(ValueError, match="Object is empty"):
        coords.azimuth
    with pytest.raises(ValueError, match="Object is empty"):
        coords.elevation
    with pytest.raises(ValueError, match="Object is empty"):
        coords.colatitude
    with pytest.raises(ValueError, match="Object is empty"):
        coords.radius
    with pytest.raises(ValueError, match="Object is empty"):
        coords.rho
    with pytest.raises(ValueError, match="Object is empty"):
        coords.phi
    with pytest.raises(ValueError, match="Object is empty"):
        coords.theta
    with pytest.raises(ValueError, match="Object is empty"):
        coords.lateral
    with pytest.raises(ValueError, match="Object is empty"):
        coords.polar


def test_setter_weights():
    """Test setting weights."""
    coords = Coordinates([1, 2], 0, 0)
    coords.weights = [.5, .5]
    assert (coords.weights == np.array([.5, .5])).all()


def test_setter_sh_order():
    """Test setting the SH order."""
    coords = Coordinates()
    coords.sh_order = 10
    assert coords.sh_order == 10


def test_setter_comment():
    """Test setting the comment."""
    coords = Coordinates()
    coords.comment = 'now this'
    assert coords.comment == 'now this'


def test_cshape():
    """Test the cshape attribute."""
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
    """Test the csim attribute."""
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
    """Test the csize attribute."""
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
    """Test getitem with different parameters."""
    # test without weights
    coords = Coordinates([1, 2], 0, 0)
    new = coords[0]
    assert isinstance(new, Coordinates)
    npt.assert_allclose(new.get_cart(), np.atleast_2d([1, 0, 0]))

    # test with weights
    coords = Coordinates([1, 2], 0, 0, weights=[.1, .9])
    new = coords[0]
    assert isinstance(new, Coordinates)
    npt.assert_allclose(new.get_cart(), np.atleast_2d([1, 0, 0]))
    assert new.weights == np.array(.1)

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


def test_find_nearest_k():
    """Test returns of find_nearest_k"""
    # 1D cartesian, nearest point
    x = np.arange(6)
    coords = Coordinates(x, 0, 0)
    i, m = coords.find_nearest_k(1, 0, 0)
    assert i == 1
    npt.assert_allclose(m, np.array([0, 1, 0, 0, 0, 0]))

    # 1D spherical, nearest point
    i, m = coords.find_nearest_k(0, 0, 1, 1, 'sph', 'top_elev', 'deg')
    assert i == 1
    npt.assert_allclose(m, np.array([0, 1, 0, 0, 0, 0]))

    # 1D cartesian, two nearest points
    i, m = coords.find_nearest_k(1.2, 0, 0, 2)
    npt.assert_allclose(i, np.array([1, 2]))
    npt.assert_allclose(m, np.array([0, 1, 1, 0, 0, 0]))

    # 1D cartesian query two points
    i, m = coords.find_nearest_k([1, 2], 0, 0)
    npt.assert_allclose(i, [1, 2])
    npt.assert_allclose(m, np.array([0, 1, 1, 0, 0, 0]))

    # 2D cartesian, nearest point
    coords = Coordinates(x.reshape(2, 3), 0, 0)
    i, m = coords.find_nearest_k(1, 0, 0)
    assert i == 1
    npt.assert_allclose(m, np.array([[0, 1, 0], [0, 0, 0]]))

    # test with plot
    coords = Coordinates(x, 0, 0)
    coords.find_nearest_k(1, 0, 0, show=True)

    # test object with a single point
    coords = Coordinates(1, 0, 0)
    coords.find_nearest_k(1, 0, 0, show=True)

    # test out of range parameters
    with raises(AssertionError):
        coords.find_nearest_k(1, 0, 0, -1)

    plt.close("all")


def test_find_nearest_cart():
    """Tests returns of find_nearest_cart."""
    # test only 1D case since most of the code from self.find_nearest_k is used
    x = np.arange(6)
    coords = Coordinates(x, 0, 0)
    i, m = coords.find_nearest_cart(2.5, 0, 0, 1.5)
    npt.assert_allclose(i, np.array([1, 2, 3, 4]))
    npt.assert_allclose(m, np.array([0, 1, 1, 1, 1, 0]))

    # test search with empty results
    i, m = coords.find_nearest_cart(2.5, 0, 0, .1)
    assert len(i) == 0
    npt.assert_allclose(m, np.array([0, 0, 0, 0, 0, 0]))

    # test out of range parameters
    with raises(AssertionError):
        coords.find_nearest_cart(1, 0, 0, -1)


def test_find_nearest_sph():
    """Tests returns of find_nearest_sph."""
    # test only 1D case since most of the code from self.find_nearest_k is used
    az = np.linspace(0, 40, 5)
    coords = Coordinates(az, 0, 1, 'sph', 'top_elev', 'deg')
    i, m = coords.find_nearest_sph(25, 0, 1, 5, 'sph', 'top_elev', 'deg')
    npt.assert_allclose(i, np.array([2, 3]))
    npt.assert_allclose(m, np.array([0, 0, 1, 1, 0]))

    # test search with empty results
    i, m = coords.find_nearest_sph(25, 0, 1, 1, 'sph', 'top_elev', 'deg')
    assert len(i) == 0
    npt.assert_allclose(m, np.array([0, 0, 0, 0, 0]))

    # test out of range parameters
    with raises(AssertionError):
        coords.find_nearest_sph(1, 0, 0, -1)
    with raises(AssertionError):
        coords.find_nearest_sph(1, 0, 0, 181)

    # test assertion for multiple radii
    coords = Coordinates([1, 2], 0, 0)
    with raises(ValueError, match="find_nearest_sph only works if"):
        coords.find_nearest_sph(0, 0, 1, 1)


def test_find_slice():
    """Test different queries for find slice."""
    # test only for self.cdim = 1.
    # self.find_slice uses KDTree, which is tested with N-dimensional arrays
    # in test_find_nearest_k()

    # cartesian grid
    d = np.linspace(-2, 2, 5)

    c = Coordinates(d, 0, 0)
    index, mask = c.find_slice('x', 'met', 0, 1)
    npt.assert_allclose(index[0], np.array([1, 2, 3]))
    npt.assert_allclose(mask, np.array([0, 1, 1, 1, 0]))

    c = Coordinates(0, d, 0)
    index, mask = c.find_slice('y', 'met', 0, 1)
    npt.assert_allclose(index[0], np.array([1, 2, 3]))
    npt.assert_allclose(mask, np.array([0, 1, 1, 1, 0]))

    c = Coordinates(0, 0, d)
    index, mask = c.find_slice('z', 'met', 0, 1)
    npt.assert_allclose(index[0], np.array([1, 2, 3]))
    npt.assert_allclose(mask, np.array([0, 1, 1, 1, 0]))

    # spherical grid
    d = [358, 359, 0, 1, 2]
    c = Coordinates(d, 0, 1, 'sph', 'top_elev', 'deg')
    # cyclic query for lower bound
    index, mask = c.find_slice('azimuth', 'deg', 0, 1)
    npt.assert_allclose(index[0], np.array([1, 2, 3]))
    npt.assert_allclose(mask, np.array([0, 1, 1, 1, 0]))
    # cyclic query for upper bound
    index, mask = c.find_slice('azimuth', 'deg', 359, 2)
    npt.assert_allclose(index[0], np.array([0, 1, 2, 3]))
    npt.assert_allclose(mask, np.array([1, 1, 1, 1, 0]))
    # non-cyclic query
    index, mask = c.find_slice('azimuth', 'deg', 1, 1)
    npt.assert_allclose(index[0], np.array([2, 3, 4]))
    npt.assert_allclose(mask, np.array([0, 0, 1, 1, 1]))
    # out of range query
    with raises(AssertionError):
        c.find_slice('azimuth', 'deg', -1, 1)
    # non existing coordinate query
    with raises(ValueError, match="'elevation' in 'ged' does not exist"):
        c.find_slice('elevation', 'ged', 1, 1)
    # test with show
    c.find_slice('azimuth', 'deg', 1, 1, show=True)

    # there is no unique processing for cylindrical coordinates - they are thus
    # not tested here.

    plt.close("all")


@pytest.mark.parametrize("rot_type,rot", [
    ('quat', [0, 0, 1 / np.sqrt(2), 1 / np.sqrt(2)]),
    ('matrix',  [[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
    ('rotvec', [0, 0, 90]),
    ('z', 90)])
def test_rotation(rot_type, rot):
    """Test rotation with different formats."""
    c = Coordinates(1, 0, 0)
    c.rotate(rot_type, rot)
    npt.assert_allclose(c.get_cart(), np.atleast_2d([0, 1, 0]), atol=1e-15)


def test_rotation_assertion():
    """Test rotation with unknown rotation type."""
    c = Coordinates(1, 0, 0)
    # test with unknown type
    with raises(ValueError):
        c.rotate('urgh', 90)


def test_inverse_rotation():
    """Test the inverse rotation."""
    xyz = np.concatenate((np.ones((2, 4, 1)),
                          np.zeros((2, 4, 1)),
                          np.zeros((2, 4, 1))), -1)
    c = Coordinates(xyz[..., 0].copy(), xyz[..., 1].copy(), xyz[..., 2].copy())
    c.rotate('z', 90)
    c.rotate('z', 90, inverse=True)
    npt.assert_allclose(c.get_cart(), xyz, atol=1e-15)


def test_converters():
    """
    Test if converterts can handle numbers (correctness of theconversion is
    tested in test_setter_and_getter_with_conversion)
    """
    coordinates.cart2sph(0, 0, 1)
    coordinates.sph2cart(0, 0, 1)
    coordinates.cart2cyl(0, 0, 1)
    coordinates.cyl2cart(0, 0, 1)


@pytest.mark.parametrize(
    'points_1, points_2, points_3, actual, expected', [
        (1, 1, 1,                Coordinates(1, 1, -1),                 False),
        (1, 1, 1,                Coordinates(),                         False),
        (1, 1, 1,                Coordinates(1, 1, 1),                  True),
        ([1, 1], [1, 1], [1, 1], Coordinates([1, 1], [1, 1], [1, 2]),   False),
        ([1, 1], [1, 1], [1, 1], Coordinates([1, 1.0], [1, 1.0], [1, 1]), True)
    ])
def test___eq___differInPoints(
        points_1, points_2, points_3, actual, expected):
    """This function checks against 3 different pairings of Coordinates."""
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


def test___eq___copy():
    coordinates = Coordinates(1, 2, 3, comment="Madre mia!")
    actual = coordinates.copy()
    assert coordinates == actual


@pytest.mark.parametrize(
    'x, y, z, radius, radius_z', [
        (1, 0, 0, 1, 1),
        (-1, 0, 0, 1, 1),
        (0, 2, 0, 2, 2),
        (0, 3, 4, 5, 3),
        (0, 0, 0, 0, 0),
    ])
def test_getter_radii_from_cart(x, y, z, radius, radius_z):
    coords = Coordinates(x, y, z)
    np.testing.assert_allclose(coords.radius, radius, atol=1e-15)
    np.testing.assert_allclose(coords.rho, radius_z, atol=1e-15)
    np.testing.assert_allclose(coords.radius, radius, atol=1e-15)


@pytest.mark.parametrize(
    'x, y, z, azimuth, elevation', [
        (1, 0, 0, 0, 0),
        (-1, 0, 0, np.pi, 0),
        (0, 1, 0, np.pi/2, 0),
        (0, -1, 0, 3*np.pi/2, 0),
        (0, 0, 1, 0, np.pi/2),
        (0, 0, -1, 0, -np.pi/2),
    ])
def test_getter_sph_top_from_cart(x, y, z, azimuth, elevation):
    coords = Coordinates(x, y, z)
    colatitude = np.pi/2 - elevation
    np.testing.assert_allclose(coords.azimuth, azimuth, atol=1e-15)
    np.testing.assert_allclose(coords.elevation, elevation, atol=1e-15)
    np.testing.assert_allclose(coords.colatitude, colatitude, atol=1e-15)
    np.testing.assert_allclose(
        coords.spherical_elevation,
        np.atleast_2d([azimuth, elevation, 1]), atol=1e-15)
    np.testing.assert_allclose(
        coords.spherical_colatitude,
        np.atleast_2d([azimuth, colatitude, 1]), atol=1e-15)
    coords = Coordinates(0, 5, 0)
    coords.azimuth = azimuth
    coords.elevation = elevation
    coords.radius = 1
    colatitude = np.pi/2 - elevation
    np.testing.assert_allclose(coords.x, x, atol=1e-15)
    np.testing.assert_allclose(coords.y, y, atol=1e-15)
    np.testing.assert_allclose(coords.z, z, atol=1e-15)


@pytest.mark.parametrize(
    'x, y, z, phi, theta', [
        (0, 1, 0, 0, np.pi/2),
        (0, -1, 0, np.pi, np.pi/2),
        (0, 0, 1, np.pi/2, np.pi/2),
        (0, 0, -1, 3*np.pi/2, np.pi/2),
        (1, 0, 0, 0, 0),
        (-1, 0, 0, 0, np.pi),
    ])
def test_getter_sph_front_from_cart(x, y, z, phi, theta):
    coords = Coordinates(x, y, z)
    np.testing.assert_allclose(coords.phi, phi, atol=1e-15)
    np.testing.assert_allclose(coords.theta, theta, atol=1e-15)
    np.testing.assert_allclose(
        coords.spherical_front, np.atleast_2d([phi, theta, 1]), atol=1e-15)
    coords = Coordinates(0, 5, 0)
    coords.phi = phi
    coords.theta = theta
    coords.radius = 1
    np.testing.assert_allclose(coords.x, x, atol=1e-15)
    np.testing.assert_allclose(coords.y, y, atol=1e-15)
    np.testing.assert_allclose(coords.z, z, atol=1e-15)


@pytest.mark.parametrize(
    'x, y, z, lateral, polar', [
        (0, 1, 0, np.pi/2, 0),
        (0, -1, 0, -np.pi/2, 0),
        (0, 0, 1, 0, np.pi/2),
        (0, 0, -1, 0, -np.pi/2),
        (1, 0, 0, 0, 0),
        (-1, 0, 0, 0, np.pi),
    ])
def test_getter_sph_side_from_cart(x, y, z, lateral, polar):
    coords = Coordinates(x, y, z)
    np.testing.assert_allclose(coords.lateral, lateral, atol=1e-15)
    np.testing.assert_allclose(coords.polar, polar, atol=1e-15)
    np.testing.assert_allclose(
        coords.spherical_side, np.atleast_2d([lateral, polar, 1]), atol=1e-15)
    coords = Coordinates(0, 5, 0)
    coords.lateral = lateral
    coords.polar = polar
    coords.radius = 1
    np.testing.assert_allclose(coords.x, x, atol=1e-15)
    np.testing.assert_allclose(coords.y, y, atol=1e-15)
    np.testing.assert_allclose(coords.z, z, atol=1e-15)


@pytest.mark.parametrize(
    'x, y, z', [
        (0, 1, 0),
        (0, -1, 0),
        (0., 0, 1),
        (0, .0, -1),
        (1, 0, 0),
        (-1, 0, 0),
        (np.ones((2, 3, 1)), np.zeros((2, 3, 1)), np.ones((2, 3, 1))),
    ])
def test_cart_setter_same_size(x, y, z):
    coords = Coordinates(x, y, z)
    np.testing.assert_allclose(coords.x, x, atol=1e-15)
    np.testing.assert_allclose(coords.y, y, atol=1e-15)
    np.testing.assert_allclose(coords.z, z, atol=1e-15)
    if x is np.array:
        np.testing.assert_allclose(
            coords.cartesian.shape[:-1], x.shape, atol=1e-15)
    np.testing.assert_allclose(coords.cartesian.shape[-1], 3, atol=1e-15)
    np.testing.assert_allclose(coords.cartesian[..., 0], coords.x, atol=1e-15)
    np.testing.assert_allclose(coords.cartesian[..., 1], coords.y, atol=1e-15)
    np.testing.assert_allclose(coords.cartesian[..., 2], coords.z, atol=1e-15)


@pytest.mark.parametrize(
    'x, y, z', [
        (np.ones((2, 3, 1)), 10, -1),
        (np.ones((2,)), 2, 1),
    ])
def test_cart_setter_different_size(x, y, z):
    coords = Coordinates(x, y, z)
    np.testing.assert_allclose(coords.x, x, atol=1e-15)
    np.testing.assert_allclose(coords.y, y, atol=1e-15)
    np.testing.assert_allclose(coords.z, z, atol=1e-15)
    if x is np.array:
        np.testing.assert_allclose(
            coords.cartesian.shape[:-1], x.shape, atol=1e-15)
    np.testing.assert_allclose(coords.cartesian.shape[-1], 3, atol=1e-15)
    np.testing.assert_allclose(coords.cartesian[..., 0], coords.x, atol=1e-15)
    np.testing.assert_allclose(coords.cartesian[..., 1], coords.y, atol=1e-15)
    np.testing.assert_allclose(coords.cartesian[..., 2], coords.z, atol=1e-15)


@pytest.mark.parametrize(
    'x, y, z', [
        (np.ones((3, 1)), 7, 3),
        (np.ones((1, 2)), 5, 1),
        (np.ones((1, 1)), 5, 1),
    ])
def test_cart_setter_different_size_with_flatten(x, y, z):
    coords = Coordinates(x, y, z)
    shape = x.flatten().shape
    np.testing.assert_allclose(coords.x, x.flatten(), atol=1e-15)
    np.testing.assert_allclose(coords.y, np.ones(shape)*y, atol=1e-15)
    np.testing.assert_allclose(coords.z, np.ones(shape)*z, atol=1e-15)
    if x is np.array:
        np.testing.assert_allclose(
            coords.cartesian.shape[:-1], x.shape, atol=1e-15)
    np.testing.assert_allclose(coords.cartesian.shape[-1], 3, atol=1e-15)


@pytest.mark.parametrize(
    'x, y, z', [
        (0, 1, 0),
        (0, -1, 0),
        (0., 0, 1),
        (0, .0, -1),
        (1, 0, 0),
        (-1, 0, 0),
        (np.ones((2, 3, 1)), 10, -1),
        (np.ones((2,)), 2, 1),
        (np.ones((2, 3, 1)), np.zeros((2, 3, 1)), np.ones((2, 3, 1))),
    ])
def test__array__getter(x, y, z):
    coords = Coordinates(x, y, z)
    np.testing.assert_allclose(
        np.array(coords)[..., 0], x, atol=1e-15)
    np.testing.assert_allclose(
        np.array(coords)[..., 1], y, atol=1e-15)
    np.testing.assert_allclose(
        np.array(coords)[..., 2], z, atol=1e-15)


@pytest.mark.parametrize(
    'x, y, z', [
        (np.ones((3, 1)), 7, 3),
        (np.ones((1, 2)), 5, 1),
        (np.ones((1, 1)), 5, 1),
    ])
def test__array__getter_with_flatten(x, y, z):
    coords = Coordinates(x, y, z)
    np.testing.assert_allclose(
        np.array(coords)[..., 0], x.flatten(), atol=1e-15)
    np.testing.assert_allclose(
        np.array(coords)[..., 1], y, atol=1e-15)
    np.testing.assert_allclose(
        np.array(coords)[..., 2], z, atol=1e-15)


def test__getitem__():
    """Test getitem with different parameters."""
    # test without weights
    coords = Coordinates([1, 2], 0, 0)
    new = coords[0]
    assert isinstance(new, Coordinates)
    np.testing.assert_allclose(new.x, 1)
    np.testing.assert_allclose(new.y, 0)
    np.testing.assert_allclose(new.z, 0)


def test__getitem__weights():
    # test with weights
    coords = Coordinates([1, 2], 0, 0, weights=[.1, .9])
    new = coords[0]
    assert isinstance(new, Coordinates)
    np.testing.assert_allclose(new.x, 1)
    np.testing.assert_allclose(new.y, 0)
    np.testing.assert_allclose(new.z, 0)
    assert new.weights == np.array(.1)


def test__getitem__3D_array():
    # test with 3D array
    coords = Coordinates([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]], 0, 0)
    new = coords[0:1]
    assert isinstance(new, Coordinates)
    assert new.cshape == (1, 5)


def test__getitem__untouced():
    # test if sliced object stays untouched
    coords = Coordinates([0, 1], [0, 1], [0, 1])
    new = coords[0]
    new.set_cart(2, 2, 2)
    assert coords.cshape == (2,)
    np.testing.assert_allclose(coords.x[0], 0)
    np.testing.assert_allclose(coords.y[0], 0)
    np.testing.assert_allclose(coords.z[0], 0)


def test__repr__comment():
    coords = Coordinates([0, 1], [0, 1], [0, 1], comment="Madre Mia!")
    x = coords.__repr__()
    assert 'Madre Mia!' in x


def test_find_slice_cart():
    """Test different queries for find slice."""
    # test only for self.cdim = 1.
    # self.find_slice uses KDTree, which is tested with N-dimensional arrays
    # in test_find_nearest_k()
    d = np.linspace(-2, 2, 5)

    c = Coordinates(d, 0, 0)
    index, mask = c.find_slice('x', 'met', 0, 1)
    np.testing.assert_allclose(index[0], np.array([1, 2, 3]))
    np.testing.assert_allclose(mask, np.array([0, 1, 1, 1, 0]))

    c = Coordinates(0, d, 0)
    index, mask = c.find_slice('y', 'met', 0, 1)
    np.testing.assert_allclose(index[0], np.array([1, 2, 3]))
    np.testing.assert_allclose(mask, np.array([0, 1, 1, 1, 0]))

    c = Coordinates(0, 0, d)
    index, mask = c.find_slice('z', 'met', 0, 1)
    np.testing.assert_allclose(index[0], np.array([1, 2, 3]))
    np.testing.assert_allclose(mask, np.array([0, 1, 1, 1, 0]))


@pytest.mark.parametrize(
    'coordinate, unit, value, tol, des_index, des_mask', [
        ('azimuth', 'deg', 0, 1, np.array([1, 2, 3]),
            np.array([0, 1, 1, 1, 0])),
        ('azimuth', 'deg', 359, 2, np.array([0, 1, 2, 3]),
            np.array([1, 1, 1, 1, 0])),
        ('azimuth', 'deg', 1, 1, np.array([2, 3, 4]),
            np.array([0, 0, 1, 1, 1])),
    ])
def test_find_slice_sph(coordinate, unit, value, tol, des_index, des_mask):
    """Test different queries for find slice."""
    # spherical grid
    d = np.array([358, 359, 0, 1, 2])
    c = Coordinates(d, 0, 1, 'sph', 'top_elev', 'deg')

    index, mask = c.find_slice(coordinate, unit, value, tol)
    np.testing.assert_allclose(index[0], des_index)
    np.testing.assert_allclose(mask, des_mask)


def test_find_slice_error():
    d = np.array([358, 359, 0, 1, 2])
    c = Coordinates(d, 0, 1, 'sph', 'top_elev', 'deg')
    # out of range query
    # with pytest.raises(AssertionError):
    #     c.find_slice('azimuth', 'deg', -1, 1)
    # non existing coordinate query
    with pytest.raises(ValueError, match="does not exist"):
        c.find_slice('elevation', 'ged', 1, 1)
    with pytest.raises(ValueError, match="does not exist"):
        c.find_slice('Ola', 'red', 1, 1)


@pytest.mark.parametrize(
    'coordinate, min, max', [
        ('azimuth', 0, 2*np.pi),
        ('polar', -np.pi/2, 3*np.pi/2),
        ('phi', 0, 2*np.pi),
    ])
def test_angle_limits_cyclic(coordinate, min, max):
    """Test different queries for find slice."""
    # spherical grid
    d = np.arange(-4*np.pi, 4*np.pi, np.pi/4)
    c = Coordinates(d, 0, 1)
    c.__setattr__(coordinate, d)
    attr = c.__getattribute__(coordinate)
    desired = (d - min) % (max - min) + min
    np.testing.assert_allclose(attr, desired, atol=2e-14)


@pytest.mark.parametrize(
    'coordinate, min, max', [
        ('azimuth', 0, 2*np.pi),
        ('polar', -np.pi/2, 3*np.pi/2),
        ('phi', 0, 2*np.pi),
        ('colatitude', 0, np.pi),
        ('theta', 0, np.pi),
        ('elevation', -np.pi/2, np.pi/2),
        ('lateral', -np.pi/2, np.pi/2),
        ('radius', 0, np.inf),
        ('rho', 0, np.inf),
    ])
def test_angle_limits(coordinate, min, max):
    """Test different queries for find slice."""
    # spherical grid
    d = np.arange(-4*np.pi, 4*np.pi, np.pi/4)
    c = Coordinates(d, 0, 1)
    c.__setattr__(coordinate, d)
    attr = c.__getattribute__(coordinate)
    assert all(attr <= max)
    assert all(attr >= min)


def test__repr__dim():
    coords = Coordinates([0, 1], [0, 1], [0, 1])
    x = coords.__repr__()
    assert '1D' in x
    assert '2' in x
    assert '(2,)' in x


@pytest.mark.parametrize(
    'coords', [
        (Coordinates(np.linspace(0, 1, 11), 0, 5)),
        (Coordinates(np.arange(10), 0, 5)),
    ])
def test_find_nearest_points_distance_1d(coords):
    for index in range(coords.csize):
        find = coords[index]
        d, i, m = coords.find_nearest_points(find, 1)
        npt.assert_array_almost_equal(d, 0)
        npt.assert_array_almost_equal(find.cartesian, coords[m].cartesian)
        npt.assert_array_almost_equal(coords[i].cartesian, find.cartesian)


@pytest.mark.parametrize(
    'coords', [
        (Coordinates(np.arange(9).reshape(3, 3), 0, 1)),
        (Coordinates(np.arange(8).reshape(2, 4), 5, 1))
    ])
def test_find_nearest_points_distance_2d(coords):
    for i in range(coords.cshape[0]):
        for j in range(coords.cshape[1]):
            find = coords[i, j]
            d, idx, m = coords.find_nearest_points(find, 1)
            npt.assert_array_almost_equal(d, 0)
            # assert find == coords[idx]
            npt.assert_array_almost_equal(find.cartesian, coords[m].cartesian)
            npt.assert_array_almost_equal(find.cartesian, coords[idx].cartesian)


def test_find_nearest_by_distance_angular():
    """Tests returns of find_nearest_sph."""
    # test only 1D case since most of the code from self.find_nearest_k is used
    az = np.linspace(0, 40, 5)
    coords = Coordinates(az, 0, 1, 'sph', 'top_elev', 'deg')
    find = Coordinates(25, 0, 1, 'sph', 'top_elev', 'deg')
    i, m = coords.find_nearest_by_distance(find, 5, 'angular')
    npt.assert_allclose(i, np.array([[2, 3]]))
    npt.assert_allclose(m, np.array([0, 0, 1, 1, 0]))

    # test search with empty results
    i, m = coords.find_nearest_by_distance(find, 1, 'angular')
    assert len(i) is 1
    assert not i[0]
    npt.assert_allclose(m, np.array([0, 0, 0, 0, 0]))


def test_find_nearest_by_distance_angular_error():
    az = np.linspace(0, 40, 5)
    coords = Coordinates(az, 0, 1, 'sph', 'top_elev', 'deg')
    # test out of range parameters
    with raises(AssertionError):
        find = Coordinates(1, 0, 0)
        coords.find_nearest_by_distance(find, -1, 'angular')
        # coords.find_nearest_sph(1, 0, 0, -1)
    with raises(AssertionError):
        find = Coordinates(1, 0, 0)
        coords.find_nearest_by_distance(find, 181, 'angular')
        # coords.find_nearest_sph(1, 0, 0, 181)

    # test assertion for multiple radii
    coords = Coordinates([1, 2], 0, 0)
    with raises(ValueError, match="find_nearest_sph only works if"):
        find = Coordinates(0, 0, 1)
        coords.find_nearest_by_distance(find, 1, 'angular')
        # coords.find_nearest_sph(0, 0, 1, 1)


def test_find_nearest_by_distance_direct():
    """Tests returns of find_nearest_cart."""
    # test only 1D case since most of the code from self.find_nearest_k is used
    x = np.arange(6)
    coords = Coordinates(x, 0, 0)
    find = Coordinates(2.5, 0, 0)
    i, m = coords.find_nearest_by_distance(find, 1.5)
    npt.assert_allclose(i, np.array([[1, 2, 3, 4]]))
    npt.assert_allclose(m, np.array([0, 1, 1, 1, 1, 0]))

    # test search with empty results
    i, m = coords.find_nearest_by_distance(find, .1)
    assert len(i) is 1
    assert not i[0]
    npt.assert_allclose(m, np.array([0, 0, 0, 0, 0, 0]))


def test_find_nearest_by_distance_direct_error():
    x = np.arange(6)
    coords = Coordinates(x, 0, 0)
    find = Coordinates(2.5, 0, 0)
    # test out of range parameters
    with raises(AssertionError):
        coords.find_nearest_by_distance(find, -1)


def test_find_nearest_points():
    """Test returns of find_nearest_k"""
    x = np.arange(6)
    coords = Coordinates(x, 0, 0)
    find = Coordinates(1, 0, 0)
    # 1D cartesian, nearest point
    d, i, m = coords.find_nearest_points(find)
    assert i[0] == 1
    npt.assert_allclose(m, np.array([0, 1, 0, 0, 0, 0]))

    # 1D spherical, nearest point
    find = Coordinates(0, 0, 1, 'sph', 'top_elev', 'deg')
    d, i, m = coords.find_nearest_points(find, 1)
    assert i[0] == 1
    npt.assert_allclose(m, np.array([0, 1, 0, 0, 0, 0]))


def test_find_nearest_2_points():
    """Test returns of find_nearest_k"""
    x = np.arange(6)
    coords = Coordinates(x, 0, 0)
    find = Coordinates(1, 0, 0)
    # 1D cartesian, two nearest points
    find = Coordinates(1.2, 0, 0)
    d, i, m = coords.find_nearest_points(find, 2)
    npt.assert_allclose(i, np.array([[1, 2]]))
    npt.assert_allclose(m, np.array([0, 1, 1, 0, 0, 0]))


def test_find_nearest_points_2():
    """Test returns of find_nearest_k"""
    x = np.arange(6)
    coords = Coordinates(x, 0, 0)
    find = Coordinates(1, 0, 0)
    # 1D cartesian query two points
    find = Coordinates([1, 2], 0, 0)
    d, i, m = coords.find_nearest_points(find)
    npt.assert_allclose(i, [[1, 2]])
    npt.assert_allclose(m, np.array([0, 1, 1, 0, 0, 0]))


def test_find_nearest_points_2d():
    """Test returns of find_nearest_k"""
    x = np.arange(6)
    # 2D cartesian, nearest point
    coords = Coordinates(x.reshape(2, 3), 0, 0)
    find = Coordinates(1, 0, 0)
    d, i, m = coords.find_nearest_points(find)
    assert i[0] == 0
    assert i[1] == 1
    npt.assert_allclose(m, np.array([[0, 1, 0], [0, 0, 0]]))
    npt.assert_almost_equal(find.cartesian, coords[i].cartesian)
    npt.assert_almost_equal(find.cartesian, coords[m].cartesian)


def test_find_nearest_points_errors():
    x = np.arange(6)
    coords = Coordinates(x, 0, 0)
    find = Coordinates(1, 0, 0)
    # test out of range parameters
    with raises(AssertionError, match='number of points must be'):
        coords.find_nearest_points(find, -1)

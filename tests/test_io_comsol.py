import numpy as np
import pytest

import os.path

from pyfar import io


def test_read_comsol_file_not_found():
    """Test error of find not found."""
    path = 'blabla.csv'
    with pytest.raises(FileNotFoundError):
        io.read_comsol_header(path)


def test_read_comsol_wrong_file_type():
    """Test read comsol file format."""
    filename = 'bla.bla'
    with pytest.raises(SyntaxError):
        io.read_comsol_header(filename)

    with pytest.raises(SyntaxError):
        io.read_comsol(filename)


@pytest.mark.parametrize("type",  ['.txt', '.dat', '.csv'])
def test_read_comsol_warning_for_db_values(type):
    path = os.path.join(os.getcwd(), 'tests', 'test_io_data', 'level_only')
    with pytest.warns(Warning, match='dB'):
        io.read_comsol(path + type)


def test_read_comsol_error_wrong_domain():
    path = os.path.join(
        os.getcwd(), 'tests', 'test_io_data', 'wrong_domain.txt')
    with pytest.raises(ValueError, match='Domain'):
        io.read_comsol(path)


@pytest.mark.parametrize("filename,expressions",  [
    ('intensity_average', ['pabe.Ix', 'pabe.Iy', 'pabe.Iz']),
    ('intensity_only', ['pabe.Ix', 'pabe.Iy', 'pabe.Iz']),
    ('intensity_parametric', ['pabe.Ix', 'pabe.Iy', 'pabe.Iz']),
    ('pressure_acceleration_parametric_time', ['actd.p_t', 'actd.a_inst']),
    ('pressure_only', ['pabe.p_t']),
    ('pressure_parametric', ['pabe.p_t']),
    ('intensity_product', ['pabe.p_t*pabe.v_inst*2/sqrt(2)']),
    ])
@pytest.mark.parametrize("type",  ['.txt', '.dat', '.csv'])
def test_read_comsol_header_expressions(filename, expressions, type):
    path = os.path.join(os.getcwd(), 'tests', 'test_io_data', filename)
    actual_expressions, _, _, _, _ = io.read_comsol_header(path + type)
    assert len(actual_expressions) == len(expressions)
    for i, exp in enumerate(expressions):
        assert actual_expressions[i] == expressions[i]


@pytest.mark.parametrize("filename,expressions_unit",  [
    ('intensity_average', ['W/m^2', 'W/m^2', 'W/m^2']),
    ('intensity_only', ['W/m^2', 'W/m^2', 'W/m^2']),
    ('intensity_parametric', ['W/m^2', 'W/m^2', 'W/m^2']),
    ('pressure_acceleration_parametric_time', ['Pa', 'm/s^2']),
    ('pressure_only', ['Pa']),
    ('pressure_parametric', ['Pa']),
    ('intensity_product', ['Pa*m/s']),
    ])
@pytest.mark.parametrize("type",  ['.txt', '.dat', '.csv'])
def test_read_comsol_header_expressions_unit(filename, expressions_unit, type):
    path = os.path.join(os.getcwd(), 'tests', 'test_io_data', filename)
    _, actual_units, _, _, _ = io.read_comsol_header(path + type)
    assert len(actual_units) == len(expressions_unit)
    for i, exp in enumerate(expressions_unit):
        assert actual_units[i] == expressions_unit[i]


@pytest.mark.parametrize("filename,parameters",  [
    ('intensity_average',
     {'theta': [0.0, 0.7854, 1.5708, 2.3562, 3.1416],
      'phi': [0., 1.5708, 3.1416, 4.7124]}),
    ('intensity_only', {}),
    ('intensity_parametric',
     {'theta': [0.0, 0.7854, 1.5708, 2.3562, 3.1416],
      'phi': [0., 1.5708, 3.1416, 4.7124]}),
    ('pressure_acceleration_parametric_time',
     {'A0': [0.5, 1., 1.5], 'f0': [50, 100, 150, 200]}),
    ('pressure_only', {}),
    ('pressure_parametric',
     {'theta': [0.0, 0.7854, 1.5708, 2.3562, 3.1416],
      'phi': [0., 1.5708, 3.1416, 4.7124]}),
    ])
@pytest.mark.parametrize("type",  ['.txt', '.dat', '.csv'])
def test_read_comsol_header_parameters(filename, parameters, type):
    path = os.path.join(os.getcwd(), 'tests', 'test_io_data', filename)
    _, _, actual_parameters, _, _ = io.read_comsol_header(path + type)
    assert parameters == actual_parameters


@pytest.mark.parametrize("filename,domain",  [
    ('intensity_average', 'freq'),
    ('intensity_only', 'freq'),
    ('intensity_parametric', 'freq'),
    ('pressure_acceleration_parametric_time', 'time'),
    ('pressure_only', 'freq'),
    ('pressure_parametric', 'freq'),
    ('intensity_product', 'freq'),
    ])
@pytest.mark.parametrize("type",  ['.txt', '.dat', '.csv'])
def test_read_comsol_header_domain(filename, domain, type):
    path = os.path.join(os.getcwd(), 'tests', 'test_io_data', filename)
    _, _, _, actual_domain, _ = io.read_comsol_header(path + type)
    assert domain == actual_domain


@pytest.mark.parametrize("filename,domain_data",  [
    ("intensity_average", [100, 500]),
    ("intensity_only", [100, 500]),
    ("intensity_parametric", [100, 500]),
    ("pressure_acceleration_parametric_time",
     [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]),
    ("pressure_only", [100, 500]),
    ("pressure_parametric", [100, 500]),
    ('intensity_product', [100, 500]),
    ])
@pytest.mark.parametrize("type",  ['.txt', '.dat', '.csv'])
def test_read_comsol_header_domain_data(filename, domain_data, type):
    path = os.path.join(os.getcwd(), 'tests', 'test_io_data', filename)
    _, _, _, _, actual_domain_data = io.read_comsol_header(path + type)
    assert domain_data == actual_domain_data


@pytest.mark.parametrize("filename",  [
    'intensity_average',
    'intensity_only',
    'intensity_parametric',
    'pressure_acceleration_parametric_time',
    'pressure_only',
    'pressure_parametric',
    'intensity_product',
    ])
@pytest.mark.parametrize("type",  ['.txt', '.dat', '.csv'])
def test_read_comsol_data_domain(filename, type):
    path = os.path.join(os.getcwd(), 'tests', 'test_io_data', filename)
    _, _, _, domain, domain_data = io.read_comsol_header(path + type)
    data = io.read_comsol(path + type)
    data = data[0] if isinstance(data, tuple) else data
    assert domain == data.domain
    if domain == 'freq':
        assert all(domain_data == data.frequencies)
    else:
        assert all(domain_data == data.times)


@pytest.mark.parametrize("filename,nodes",  [
    ('intensity_average', 1),
    ('intensity_only', 8),
    ('intensity_parametric', 8),
    ('pressure_acceleration_parametric_time', 46),
    ('pressure_only', 8),
    ('pressure_parametric', 8),
    ('intensity_product', 8),
    ])
@pytest.mark.parametrize("type",  ['.txt', '.dat', '.csv'])
def test_read_comsol_data_shapes(filename, nodes, type):
    path = os.path.join(os.getcwd(), 'tests', 'test_io_data', filename)
    expressions, _, parameters, _, _ = io.read_comsol_header(path + type)
    data = io.read_comsol(path + type)
    coordinates = data[1] if isinstance(data, tuple) else None
    data = data[0] if isinstance(data, tuple) else data
    if coordinates is not None:
        assert data.cshape[0] == coordinates.cshape[0]
    assert data.cshape[0] == nodes
    assert data.cshape[1] == len(expressions)
    for i, para in enumerate(parameters):
        assert data.cshape[2+i] == len(parameters[para])


@pytest.mark.parametrize("filename",  [
    'intensity_only',
    'intensity_parametric',
    'pressure_only',
    'pressure_parametric',
    'intensity_product',
    ])
@pytest.mark.parametrize("type",  ['.txt', '.dat', '.csv'])
def test_read_comsol_coordinates(filename, type):
    path = os.path.join(os.getcwd(), 'tests', 'test_io_data', filename)
    _, coordinates = io.read_comsol(path + type)
    xyz = coordinates.get_cart()
    assert all(xyz[0] == [-.5, -.5, -.5])
    assert all(xyz[1] == [0.5, -0.5, -0.5])
    assert all(xyz[2] == [-0.5, 0.5, -0.5])
    assert all(xyz[3] == [0.5, 0.5, -0.5])
    assert all(xyz[4] == [-0.5, -0.5, 0.5])
    assert all(xyz[5] == [0.5, -0.5, 0.5])
    assert all(xyz[6] == [-0.5, 0.5, 0.5])
    assert all(xyz[7] == [0.5, 0.5, 0.5])


@pytest.mark.parametrize("filename,p1",  [
    ('intensity_average', np.float64(1.2867253973047096E-24)),
    ('intensity_only', np.float64(-1.0535540378988276E-8)),
    ('intensity_parametric', np.float64(-1.0535540378988267E-8)),
    ('pressure_acceleration_parametric_time',
     np.float64(2.2469262686436532E-14)),
    ('pressure_only',
     np.complex128(complex(-3.308489057665816E-5, -0.003883799752478906))),
    ('pressure_parametric',
     np.complex128(complex(-3.3084890576658145E-5, -0.003883799752478905))),
    ('intensity_product',
     np.complex128(-2.810452490679023E-10-3.299159978287418E-8j)),
    ])
@pytest.mark.parametrize("suffix",  ['.txt', '.dat', '.csv'])
def test_read_comsol_first_value_data(filename, p1, suffix):
    path = os.path.join(os.getcwd(), 'tests', 'test_io_data', filename)
    _, _, _, domain, _ = io.read_comsol_header(path + suffix)
    data = io.read_comsol(path + suffix)
    data = data[0] if isinstance(data, tuple) else data
    if domain == 'freq':
        assert data.freq.flatten()[0] == p1
        assert type(data.freq.flatten()[0]) == type(p1)
        # returns error due to a bug in FrequencyData
    else:
        assert data.time.flatten()[0] == p1
        assert type(data.time.flatten()[0]) == type(p1)


@pytest.mark.parametrize("filename,p1",  [
    ('intensity_parametric',
     -1.0533947432793473E-8),
    ('pressure_parametric',
     complex(-1.6519703918208651E-4, -0.003880099599610891)),
    ])
@pytest.mark.parametrize("type",  ['.txt', '.dat', '.csv'])
def test_read_comsol_another_value_data(filename, p1, type):
    """Test freq=500; theta=0.7854; phi=3.1416"""
    path = os.path.join(os.getcwd(), 'tests', 'test_io_data', filename)
    data, _ = io.read_comsol(path + type)
    point = 0
    exp = 0
    freq = 1
    theta = 1
    phi = 2
    assert data.freq[point, exp, theta, phi, freq] == p1


@pytest.mark.parametrize("filename,p1",  [
    ('intensity_parametric',
     -1.0533947432793473E-8),
    ('pressure_parametric',
     complex(-1.6519703918208651E-4, -0.003880099599610891)),
    ])
@pytest.mark.parametrize("type",  ['.txt', '.dat', '.csv'])
def test_read_comsol_parameters_another_value_data(filename, p1, type):
    """Test freq=500; theta=0.7854; phi=3.1416"""
    path = os.path.join(os.getcwd(), 'tests', 'test_io_data', filename)
    _, _, parameters, _, _ = io.read_comsol_header(path + type)
    theta = 1
    phi = 2
    parameters['theta'] = [parameters['theta'][theta]]
    parameters['phi'] = [parameters['phi'][phi]]
    data, _ = io.read_comsol(path + type, parameters=parameters)
    point = 0
    exp = 0
    freq = 1
    assert data.freq[point, exp, 0, 0, freq] == p1


@pytest.mark.parametrize("filename",  [
    'intensity_average',
    'intensity_only',
    'intensity_parametric',
    'pressure_acceleration_parametric_time',
    'pressure_only',
    'pressure_parametric',
    'intensity_product'
    ])
@pytest.mark.parametrize("type",  ['.txt', '.dat', '.csv'])
def test_read_comsol_expressions(filename, type):
    path = os.path.join(os.getcwd(), 'tests', 'test_io_data', filename)
    expressions, _, _, _, _ = io.read_comsol_header(path + type)
    data = io.read_comsol(path + type, expressions=[expressions[0]])
    data = data[0] if isinstance(data, tuple) else data
    assert data.cshape[1] == 1


@pytest.mark.parametrize("filename",  [
    'pressure_parametric_incomplete',
    ])
@pytest.mark.parametrize("type",  ['.txt', '.dat', '.csv'])
def test_read_comsol_check_incomplete(filename, type):
    path = os.path.join(os.getcwd(), 'tests', 'test_io_data', filename)
    expressions, _, _, _, _ = io.read_comsol_header(path + type)
    with pytest.warns(Warning, match='Specific'):
        data, _ = io.read_comsol(
            path + type, expressions=[expressions[0]])
    assert all(np.isnan(data.freq[:, 0, -1, -1, 1]).flatten())
    data.freq[:, 0, -1, -1, 1] = 0
    assert any(~np.isnan(data.freq.flatten()))


@pytest.mark.parametrize("filename",  [
    'intensity_parametric',
    'intensity_average',
    ])
@pytest.mark.parametrize("type",  ['.txt', '.dat', '.csv'])
def test_read_comsol_parameters_shape(filename, type):
    path = os.path.join(os.getcwd(), 'tests', 'test_io_data', filename)
    expressions, expressions_unit, parameters, domain, domain_data \
        = io.read_comsol_header(path + type)
    parameters['theta'] = parameters['theta'][:1]
    parameters['phi'] = parameters['phi'][:2]
    data = io.read_comsol(path + type, parameters=parameters)
    data = data[0] if isinstance(data, tuple) else data
    assert data.cshape[1] == len(expressions)
    assert data.cshape[2] == len(parameters['theta'][:1])
    assert data.cshape[3] == len(parameters['phi'][:2])
    assert data.n_bins == 2


@pytest.mark.parametrize("filename",  [
    'intensity_parametric',
    'intensity_average',
    ])
@pytest.mark.parametrize("type",  ['.txt', '.dat', '.csv'])
def test_read_comsol_expressions_value(filename, type):
    path = os.path.join(os.getcwd(), 'tests', 'test_io_data', filename)
    expressions, expressions_unit, parameters, domain, domain_data \
        = io.read_comsol_header(path + type)
    data_exp = io.read_comsol(
        path + type, expressions=[expressions[0], expressions[2]])
    data_exp = data_exp[0] if isinstance(data_exp, tuple) else data_exp
    data = io.read_comsol(path + type)
    data = data[0] if isinstance(data, tuple) else data
    assert data_exp.freq[0, 0, 1, 1, 1] == data.freq[0, 0, 1, 1, 1]
    assert data_exp.freq[0, 1, 1, 1, 1] == data.freq[0, 2, 1, 1, 1]
    assert data_exp.freq[0, 0, 0, 2, 1] == data.freq[0, 0, 0, 2, 1]
    assert data_exp.freq[0, 1, 0, 2, 1] == data.freq[0, 2, 0, 2, 1]


@pytest.mark.parametrize("filename",  [
    'intensity_average_specific',
    'intensity_parametric_specific',
    'pressure_parametric_specific',
    ])
@pytest.mark.parametrize("type",  ['.txt', '.dat', '.csv'])
def test_read_comsol_check_specific_combination(filename, type):
    path = os.path.join(os.getcwd(), 'tests', 'test_io_data', filename)
    expressions, _, _, _, _ = io.read_comsol_header(path + type)
    with pytest.warns(Warning, match='Specific'):
        data = io.read_comsol(path + type, expressions=[expressions[0]])
    data = data[0] if isinstance(data, tuple) else data
    # For specific combinations shape of parameters need to be same
    assert data.cshape[2] == data.cshape[3]
    # test if the right values are nan and not nan
    for i in range(data.cshape[2]):
        assert all(~np.isnan(data.freq[:, :, i, i, :]).flatten())
        data.freq[:, :, i, i, :] = np.nan
    assert all(np.isnan(data.freq.flatten()))

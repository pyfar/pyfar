import pyfar.io._codec as codec


def test__str_to_type():
    """ Test if str_to_type works properly. """
    PyfarType = codec._str_to_type('Coordinates')
    assert PyfarType.__module__.startswith('pyfar')
    PyfarType = codec._str_to_type('Orientations')
    assert PyfarType.__module__.startswith('pyfar')
    PyfarType = codec._str_to_type('SphericalVoronoi')
    assert PyfarType.__module__.startswith('pyfar')

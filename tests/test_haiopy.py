import numpy.testing as npt     # noqa


def test_import_haiopy():
    try:
        import haiopy           # noqa
    except ImportError:
        assert False

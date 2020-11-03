import numpy as np
import numpy.testing as npt     # noqa
from pytest import raises


def test_import_haiopy():
    try:
        import haiopy           # noqa
    except ImportError:
        assert False

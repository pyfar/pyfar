import numpy as np
import numpy.testing as npt
import pytest


def test_import_haiopy():
    try:
        import haiopy
    except ImportError:
        assert False

import pytest


def test_import_pyfar():
    try:
        import pyfar           # noqa
    except ImportError:
        pytest.fail('import pyfar failed')

def test_import_pyfar():
    try:
        import pyfar           # noqa
    except ImportError:
        raise AssertionError()

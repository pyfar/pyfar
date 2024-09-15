def test_import_pyfar():
    try:
        import pyfar           # noqa
    except ImportError as exc:
        raise AssertionError() from exc

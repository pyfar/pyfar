import pyfar as pf

def test_ticker_module_existence():
    """Test if the ticker module is available."""
    assert hasattr(pf.plot, 'ticker')

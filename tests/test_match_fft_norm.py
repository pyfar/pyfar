import pytest
from pyfar.classes.audio import _match_fft_norm


def test_input_fft_norm_valid():
    """Test assertion by passing invalid fft_norm"""
    with pytest.raises(ValueError,
                       match='fft_norm_1 is invalid but must be in '):
        _match_fft_norm('invalid', 'none')
    with pytest.raises(ValueError,
                       match='fft_norm_2 is invalid but must be in '):
        _match_fft_norm('none', 'invalid')


def test_input_division_type():
    """Test assertion by passing invalid division parameter"""
    with pytest.raises(TypeError,
                       match='Parameter division must be type bool.'):
        _match_fft_norm('none', 'none', 'invalid')


@pytest.mark.parametrize("fft_norm_1, fft_norm_2, result",
                         [['none',       'none',         'none'],
                          ['none',       'unitary',      'unitary'],
                          ['none',       'amplitude',    'amplitude'],
                          ['none',       'power',        'power'],
                          ['none',       'psd',          'psd'],
                          ['unitary',    'none',         'unitary'],
                          ['amplitude',  'none',         'amplitude'],
                          ['rms',        'none',         'rms'],
                          ['power',      'none',         'power'],
                          ['psd',        'none',         'psd'],
                          ['unitary',    'unitary',      'unitary'],
                          ['amplitude',  'amplitude',    'amplitude'],
                          ['rms',        'rms',          'rms'],
                          ['power',      'power',        'power'],
                          ['psd',        'psd',          'psd']])
def test_result_no_division(fft_norm_1, fft_norm_2, result):
    """Test the returned fft_norm for arithmetic operation by passing
    valid combinations of fft_norms"""
    assert _match_fft_norm(fft_norm_1, fft_norm_2) == result


@pytest.mark.parametrize("fft_norm_1, fft_norm_2",
                         [['unitary',    'amplitude'],
                          ['unitary',    'rms'],
                          ['unitary',    'power'],
                          ['unitary',    'psd'],
                          ['amplitude',  'unitary'],
                          ['amplitude',  'rms'],
                          ['amplitude',  'power'],
                          ['amplitude',  'psd'],
                          ['rms',        'unitary'],
                          ['rms',        'amplitude'],
                          ['rms',        'power'],
                          ['rms',        'psd'],
                          ['power',      'unitary'],
                          ['power',      'amplitude'],
                          ['power',      'rms'],
                          ['power',      'psd'],
                          ['psd',        'unitary'],
                          ['psd',        'amplitude'],
                          ['psd',        'rms'],
                          ['psd',        'power']])
def test_assertion_no_division(fft_norm_1, fft_norm_2):
    """Test assertion by passing invalid combinations of fft_norms"""
    with pytest.raises(ValueError, match="Either one fft_norm has to be "):
        _match_fft_norm(fft_norm_1, fft_norm_2)


@pytest.mark.parametrize("fft_norm_1, fft_norm_2, result",
                         [['none',       'none',         'none'],
                          ['unitary',    'none',         'unitary'],
                          ['amplitude',  'none',         'amplitude'],
                          ['rms',        'none',         'rms'],
                          ['power',      'none',         'power'],
                          ['psd',        'none',         'psd'],
                          ['unitary',    'unitary',      'none'],
                          ['amplitude',  'amplitude',    'none'],
                          ['rms',        'rms',          'none'],
                          ['power',      'power',        'none'],
                          ['psd',        'psd',          'none']])
def test_result_division(fft_norm_1, fft_norm_2, result):
    """Test the returned fft_norm for arithmetic operation by passing
    valid combinations of fft_norms, with division=True"""
    assert _match_fft_norm(fft_norm_1, fft_norm_2, division=True) == result


@pytest.mark.parametrize("fft_norm_1, fft_norm_2",
                         [['none',       'unitary'],
                          ['none',       'amplitude'],
                          ['none',       'power'],
                          ['none',       'psd'],
                          ['unitary',    'amplitude'],
                          ['unitary',    'rms'],
                          ['unitary',    'power'],
                          ['unitary',    'psd'],
                          ['amplitude',  'unitary'],
                          ['amplitude',  'rms'],
                          ['amplitude',  'power'],
                          ['amplitude',  'psd'],
                          ['rms',        'unitary'],
                          ['rms',        'amplitude'],
                          ['rms',        'power'],
                          ['rms',        'psd'],
                          ['power',      'unitary'],
                          ['power',      'amplitude'],
                          ['power',      'rms'],
                          ['power',      'psd'],
                          ['psd',        'unitary'],
                          ['psd',        'amplitude'],
                          ['psd',        'rms'],
                          ['psd',        'power']])
def test_assertion_division(fft_norm_1, fft_norm_2):
    """Test assertion by passing invalid combinations of fft_norms,
    with division=True"""
    with pytest.raises(ValueError, match="Either fft_norm_2 "):
        _match_fft_norm(fft_norm_1, fft_norm_2, division=True)

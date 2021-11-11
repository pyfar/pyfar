import pytest
from pyfar.classes.audio import match_fft_norm


def test_input_fft_norm_valid():
    """Test assertion by passing invalid fft_norm"""
    with pytest.raises(ValueError,
                       match='Parameter fft_norm_1 is not a valid fft_norm.'):
        match_fft_norm('invalid', 'none')
    with pytest.raises(ValueError,
                       match='Parameter fft_norm_2 is not a valid fft_norm.'):
        match_fft_norm('none', 'invalid')


def test_input_division_type():
    """Test assertion by passing invalid division parameter"""
    with pytest.raises(TypeError,
                       match='Parameter division must be type bool.'):
        match_fft_norm('none', 'none', 'invalid')


def test_result_no_division():
    """Test the returned fft_norm for arithmetic operation by passing
    valid combinations of fft_norms"""
    #           fft_norm_1, fft_norm_2, result
    fft_norms = [['none', 'none', 'none'],
                 ['none', 'unitary', 'unitary'],
                 ['none', 'amplitude', 'amplitude'],
                 ['none', 'power', 'power'],
                 ['none', 'psd', 'psd'],
                 ['unitary', 'none', 'unitary'],
                 ['amplitude', 'none', 'amplitude'],
                 ['rms', 'none', 'rms'],
                 ['power', 'none', 'power'],
                 ['psd', 'none', 'psd'],
                 ['unitary', 'unitary', 'unitary'],
                 ['amplitude', 'amplitude', 'amplitude'],
                 ['rms', 'rms', 'rms'],
                 ['power', 'power', 'power'],
                 ['psd', 'psd', 'psd']]
    for fft_norm in fft_norms:
        assert match_fft_norm(fft_norm[0], fft_norm[1]) == fft_norm[2]


def test_assertion_no_division():
    """Test assertion by passing invalid combinations of fft_norms"""
    #           fft_norm_1, fft_norm_2
    fft_norms = [['unitary', 'amplitude'],
                 ['unitary', 'rms'],
                 ['unitary', 'power'],
                 ['unitary', 'psd'],
                 ['amplitude', 'unitary'],
                 ['amplitude', 'rms'],
                 ['amplitude', 'power'],
                 ['amplitude', 'psd'],
                 ['rms', 'unitary'],
                 ['rms', 'amplitude'],
                 ['rms', 'power'],
                 ['rms', 'psd'],
                 ['power', 'unitary'],
                 ['power', 'amplitude'],
                 ['power', 'rms'],
                 ['power', 'psd'],
                 ['psd', 'unitary'],
                 ['psd', 'amplitude'],
                 ['psd', 'rms'],
                 ['psd', 'power']]
    for fft_norm in fft_norms:
        with pytest.raises(ValueError,
                           match="Either one fft_norm has to be "):
            match_fft_norm(fft_norm[0], fft_norm[1])


def test_result_division():
    """Test the returned fft_norm for arithmetic operation by passing
    valid combinations of fft_norms, with division=True"""
    pass


def test_assertion_division():
    """Test assertion by passing invalid combinations of fft_norms,
    with division=True"""
    pass

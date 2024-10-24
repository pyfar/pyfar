import numpy as np
import pytest
from pyfar.classes.audio import _Audio
import pyfar as pf


def test_audio_init_with_defaults():
    """Test to init Audio without optional parameters."""
    audio = _Audio(domain='time')
    assert isinstance(audio, _Audio)
    assert audio.domain == 'time'
    assert audio.comment == ''


def test_audio_init_invalid_domain():
    with pytest.raises(ValueError, match="Incorrect domain"):
        _Audio(domain='space')


def test_audio_comment():
    audio = _Audio(domain='time', comment='Bla')
    assert audio.comment == 'Bla'

    audio.comment = 'Blub'
    assert audio.comment == 'Blub'

    with pytest.raises(TypeError, match="comment has to be of type string."):
        pf.Signal([1, 2, 3], 44100, comment=[1, 2, 3])


def test_check_input_type_is_numeric_error():
    """Test if error is raised."""
    with pytest.raises(TypeError, match="int, uint, float, or complex"):
        _Audio._check_input_type_is_numeric(np.array(['1', '2', '3']))


@pytest.mark.parametrize('dtype', ['int', 'uint', 'float', 'complex'])
def test_check_input_type_is_numeric_no_error(dtype):
    """Test if data passes as expected."""
    _Audio._check_input_type_is_numeric(np.array([1, 2, 3], dtype=dtype))


@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
def test_check_input_values_are_numeric_error(value):
    """Test if errors are raised."""
    with pytest.raises(ValueError, match="input values must be numeric"):
        _Audio._check_input_values_are_numeric(np.array([1, 2, value]))


def test_check_input_values_are_numeric_no_error():
    """Test if data passes as expected."""
    _Audio._check_input_values_are_numeric(np.array([1, 2, 3]))


def test_not_implemented():
    audio = _Audio(domain='time')

    with pytest.raises(NotImplementedError):
        audio._return_item()

    with pytest.raises(NotImplementedError):
        audio._assert_matching_meta_data()

    with pytest.raises(NotImplementedError):
        audio._decode()

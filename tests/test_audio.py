import pytest
import numpy as np
import numpy.testing as npt
from pyfar.classes.audio import _Audio


def test_audio_init_with_defaults():
    """Test to init Audio without optional parameters."""
    audio = _Audio(data=[1, 2, 3], domain='time')
    assert isinstance(audio, _Audio)
    assert audio.domain == 'time'
    npt.assert_equal(audio._data, np.atleast_2d([1, 2, 3]))


def test_audio_init_invalid_domain():
    with pytest.raises(ValueError):
        _Audio([1, 2, 3], domain='space')


def test_audio_comment():
    audio = _Audio([1, 2, 3], domain='time', comment='Bla')
    assert audio.comment == 'Bla'

    audio.comment = 'Blub'
    assert audio.comment == 'Blub'


@pytest.mark.parametrize('dtype,str_dtype', (
    [float, "float64"], [complex, "complex128"]))
def test_audio_data_type(dtype, str_dtype):
    """Test different dtypes"""

    audio = _Audio([1, 2, 3], domain="time", dtype=dtype)
    assert audio.dtype == dtype
    assert str(audio._data.dtype) == str_dtype


def test_audio_dtype_casting():
    """Test automatic casting from int to float"""

    data = np.array([1, 2, 3], dtype=int)
    audio = _Audio(data, domain="time")

    assert audio.dtype == float
    assert str(audio._data.dtype).startswith("float")
    assert str(data.dtype).startswith("int")


def test_audio_dtype_assertion():
    """Test assertions for dtype"""

    # invalid input argument for dtype
    with pytest.raises(ValueError, match="dtype must be None, float"):
        _Audio([1, 2, 3], "time", dtype=int)

    # wrong dtype of data
    with pytest.raises(ValueError, match="data is of type"):
        _Audio(["1", "2", "3"], "time")


def test_return_item():
    audio = _Audio([1, 2, 3], domain='time')

    with pytest.raises(NotImplementedError):
        audio._return_item()


def test_assert_matching_meta_data():
    audio = _Audio([1, 2, 3], domain='time')

    with pytest.raises(NotImplementedError):
        audio._assert_matching_meta_data()

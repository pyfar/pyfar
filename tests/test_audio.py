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


def test_return_item():
    audio = _Audio([1, 2, 3], domain='time')

    with pytest.raises(NotImplementedError):
        audio._return_item()


def test_assert_matching_meta_data():
    audio = _Audio([1, 2, 3], domain='time')

    with pytest.raises(NotImplementedError):
        audio._assert_matching_meta_data()

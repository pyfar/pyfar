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


def test_not_implemented():
    audio = _Audio(domain='time')

    with pytest.raises(NotImplementedError):
        audio._return_item()

    with pytest.raises(NotImplementedError):
        audio._assert_matching_meta_data()

    with pytest.raises(NotImplementedError):
        audio._decode()

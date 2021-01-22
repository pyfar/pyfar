import pytest
from pyfar.signal import _Audio


def test_audio_init_with_defaults():
    """Test to init Audio without optional parameters."""
    audio = _Audio(domain='time')
    assert isinstance(audio, _Audio)
    assert audio.domain == 'time'


def test_audio_init_invalid_domain():
    with pytest.raises(ValueError):
        _Audio(domain='space')


def test_audio_comment():
    audio = _Audio(domain='time', comment='Bla')
    assert audio.comment == 'Bla'

    audio.comment = 'Blub'
    assert audio.comment == 'Blub'

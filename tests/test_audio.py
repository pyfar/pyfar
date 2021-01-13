import pytest
from pyfar.signal import Audio as Audio


def test_audio_init_with_defaults():
    """Test to init Audio without optional parameters."""
    audio = Audio(domain='time')
    assert isinstance(audio, Audio)
    assert audio.domain == 'time'


def test_audio_init_invalid_domain():
    with pytest.raises(ValueError):
        Audio(domain='space')


def test_audio_comment():
    audio = Audio(domain='time', comment='Bla')
    assert audio.comment == 'Bla'

    audio.comment = 'Blub'
    assert audio.comment == 'Blub'

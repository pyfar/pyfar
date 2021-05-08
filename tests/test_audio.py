import pytest
from pyfar.classes.audio import _Audio


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


def test_return_item():
    audio = _Audio(domain='time')

    with pytest.raises(NotImplementedError):
        audio._return_item()


def test_assert_matching_meta_data():
    audio = _Audio(domain='time')

    with pytest.raises(NotImplementedError):
        audio._assert_matching_meta_data()

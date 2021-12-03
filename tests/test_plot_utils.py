import pyfar.plot.utils as utils
import pytest


def test_color():
    assert utils.color('r') == '#D83C27'
    assert utils.color('red') == '#D83C27'
    assert utils.color(1) == '#D83C27'
    assert utils.color(9) == '#D83C27'

    with pytest.raises(ValueError, match="color is"):
        utils.color('a')


def test_shortcuts():
    shorts = utils.shortcuts()
    assert isinstance(shorts, dict)

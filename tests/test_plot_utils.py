import haiopy.plot.utils as utils
import pytest


def test_color():
    assert utils.color('r') == '#D83C27'

    with pytest.raises(ValueError):
        utils.color('a')

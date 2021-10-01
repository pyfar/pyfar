import pyfar.plot.utils as utils
import pytest
import pyfar.plot._line as _line
import matplotlib.pyplot as plt


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


def test_default_colors():
    """Test default colors in plotstyles to match
    function used for displaying these
    """
    color_dict = _line._default_color_dict()
    colors = list(color_dict.values())
    for style in ['light', 'dark']:
        with utils.context(style):
            prop_cycle = plt.rcParams['axes.prop_cycle']
            colors_style = prop_cycle.by_key()['color']
            assert colors == colors_style

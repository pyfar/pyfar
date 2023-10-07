import pyfar.plot.utils as utils
import pytest


def test_color():
    assert utils.color('r') == '#D83C27'
    assert utils.color('red') == '#D83C27'
    assert utils.color(1) == '#D83C27'
    assert utils.color(9) == '#D83C27'

    with pytest.raises(ValueError, match="color is"):
        utils.color('a')


def test_shortcuts_console_output(capfd):
    """Test output to the console"""
    _ = utils.shortcuts(show=True)
    out, _ = capfd.readouterr()
    assert "Use these shortcuts" in out

    _ = utils.shortcuts(show=False)
    out, _ = capfd.readouterr()
    assert out == ""


def test_shortcuts_returns():
    """Test return values"""
    shorts = utils.shortcuts(show=False)
    assert isinstance(shorts, dict)

    shorts, report = utils.shortcuts(show=False, report=True)
    assert isinstance(shorts, dict)
    assert isinstance(report, str)


def test_shortcuts_report():
    """Test layout of return / console output"""
    _, report = utils.shortcuts(show=False, report=True, layout="console")
    assert ".. list-table::" not in report

    _, report = utils.shortcuts(show=False, report=True, layout="sphinx")
    assert ".. list-table::" in report

    with pytest.raises(ValueError, match="layout is 'tex'"):
        utils.shortcuts(report=True, layout="tex")

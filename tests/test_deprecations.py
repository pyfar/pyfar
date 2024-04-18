"""
Test deprecations. For each deprecation two things must be tested:
1. Is a proper warning raised. This is done using
   with pytest.warns(PyfarDeprecationWarning, match="some text"):
       call_of_function()
2. Was the function properly deprecated. This is done using:
   if version.parse(pf.__version__) >= version.parse('0.5.0'):
        with pytest.raises(AttributeError):
            # remove get_nearest_k() from pyfar 0.5.0!
            coords.get_nearest_k(1, 0, 0)

"""
import numpy as np
from packaging import version
import re

import pytest

import pyfar as pf
import pyfar.signals as pfs
from pyfar.classes.warnings import PyfarDeprecationWarning

# This defines the plot size and the backend
from pyfar.testing.plot_utils import create_figure


# deprecate in 0.6.0 ----------------------------------------------------------
@pytest.mark.parametrize('function', [
    (pf.plot.freq), (pf.plot.phase), (pf.plot.group_delay),
    (pf.plot.time_freq), (pf.plot.freq_phase), (pf.plot.freq_group_delay)])
def test_xscale_deprecation(function, handsome_signal):
    """Deprecate xscale parameter in plot functions"""

    if version.parse(pf.__version__) >= version.parse('0.6.0'):
        with pytest.raises(AttributeError):
            # remove xscale from pyfar 0.6.0!
            create_figure()
            function(handsome_signal, xscale='linear')


def test_spectrogram_yscale_deprecation(sine):
    """Deprecate yscale parameter in plot functions"""

    if version.parse(pf.__version__) >= version.parse('0.6.0'):
        with pytest.raises(AttributeError):
            # remove yscale from pyfar 0.6.0!
            create_figure()
            pf.plot.spectrogram(sine, yscale='linear')


def test__check_time_unit():
    """Deprecate unit=None in plots showing the time or group delay"""

    if version.parse(pf.__version__) >= version.parse('0.6.0'):
        with pytest.raises(ValueError):
            # remove xscale from pyfar 0.6.0!
            create_figure()
            pf.plot._utils._check_time_unit(None)


# deprecate in 0.8.0 ----------------------------------------------------------
def test_pad_zero_modi():
    with pytest.warns(PyfarDeprecationWarning,
                      match='Mode "before" and "after" will be renamed into'):
        pf.dsp.pad_zeros(pf.Signal([1], 44100), 5, 'before')

    if version.parse(pf.__version__) >= version.parse('0.8.0'):
        with pytest.raises(ValueError):
            # remove mode 'before' and 'after' from pyfar 0.8.0!
            pf.dsp.pad_zeros(pf.Signal([1], 44100), 5, mode='before')


# deprecate in 0.8.0 ----------------------------------------------------------
@pytest.mark.parametrize(
    'statement', [
        ('coords.get_cart()'),
        ('coords.sh_order'),
        ('coords.set_cart(1,1,1)'),
        ('coords.get_cyl()'),
        ('coords.set_cyl(1,1,1)'),
        ('coords.get_sph()'),
        ('coords.set_sph(1,1,1)'),
        ('coords.systems()'),
        ('pf.Coordinates(0, 0, 0, sh_order=1)'),
        ("pf.Coordinates(0, 0, 0, domain='sph')"),
        ("pf.Coordinates(0, 0, 0, domain='sph', unit='deg')"),
        ("pf.Coordinates(0, 0, 0, domain='sph', convention='top_colat')"),
        ("pf.samplings.cart_equidistant_cube(2)"),
        ("pf.samplings.sph_dodecahedron()"),
        ("pf.samplings.sph_icosahedron()"),
        ("pf.samplings.sph_equiangular(sh_order=5)"),
        ("pf.samplings.sph_gaussian(sh_order=5)"),
        ("pf.samplings.sph_extremal(sh_order=5)"),
        ("pf.samplings.sph_t_design(sh_order=5)"),
        ("pf.samplings.sph_equal_angle(5)"),
        ("pf.samplings.sph_great_circle()"),
        ("pf.samplings.sph_lebedev(sh_order=5)"),
        ("pf.samplings.sph_fliege(sh_order=5)"),
        ("pf.samplings.sph_equal_area(5)"),
    ])
def test_deprecations_0_8_0(statement):
    coords = pf.Coordinates.from_spherical_colatitude(np.arange(6), 0, 0)
    coords.y = 1

    # PyfarDeprecationWarning for
    with pytest.warns(PyfarDeprecationWarning,
                      match="This function will be"):
        eval(statement)

    # PyfarDeprecationWarning check version
    with pytest.warns(PyfarDeprecationWarning,
                      match="0.8.0"):
        eval(statement)

    # remove statement from pyfar 0.8.0!
    if version.parse(pf.__version__) >= version.parse('0.8.0'):
        with pytest.raises(AttributeError):
            eval(statement)


def test_deprecations_0_8_0_set_sh_order():
    coords = pf.Coordinates(np.arange(6), 0, 0)
    # sh_order setter
    with pytest.warns(PyfarDeprecationWarning,
                      match="This function will be deprecated"):
        coords.sh_order = 1

    # sh_order setter
    with pytest.warns(PyfarDeprecationWarning,
                      match="0.8.0"):
        coords.sh_order = 1

    # remove statement from pyfar 0.8.0!
    if version.parse(pf.__version__) >= version.parse('0.8.0'):
        with pytest.raises(AttributeError):
            coords.sh_order = 1


def test_signal_len():
    with pytest.warns(PyfarDeprecationWarning,
                      match=re.escape("len(Signal) will be deprecated")):
        len(pf.Signal([1, 2, 3], 44100))

    if version.parse(pf.__version__) >= version.parse('0.8.0'):
        with pytest.raises(TypeError, match=re.escape("had no len()")):
            # remove Signal.__len__ from pyfar 0.8.0!
            len(pf.Signal([1, 2, 3], 44100))


def test_deprecations_find_nearest_k():
    coords = pf.Coordinates(np.arange(6), 0, 0)

    with pytest.warns(
            PyfarDeprecationWarning,
            match="This function will be deprecated in pyfar 0.8.0 in favor"):
        coords.find_nearest_k(1, 0, 0)

    if version.parse(pf.__version__) >= version.parse('0.8.0'):
        with pytest.raises(TypeError):
            coords.find_nearest_k(1, 0, 0)


def test_deprecations_find_slice():
    coords = pf.samplings.sph_lebedev(sh_order=10)

    with pytest.warns(
            PyfarDeprecationWarning,
            match="This function will be deprecated in pyfar 0.8.0. Use "):
        coords.find_slice('elevation', 'deg', 0, 5)

    if version.parse(pf.__version__) >= version.parse('0.8.0'):
        with pytest.raises(TypeError):
            coords.find_slice('elevation', 'deg', 0, 5)


def test_deprecations_find_nearest_cart():
    coords = pf.samplings.sph_lebedev(sh_order=10)

    with pytest.warns(
            PyfarDeprecationWarning,
            match="This function will be deprecated in pyfar 0.8.0 in favor "):
        coords.find_nearest_cart(1, 1, 1, 1)

    if version.parse(pf.__version__) >= version.parse('0.8.0'):
        with pytest.raises(TypeError):
            coords.find_nearest_cart(1, 1, 1, 1)


def test_deprecations_find_nearest_sph():
    coords = pf.samplings.sph_lebedev(sh_order=10)

    with pytest.warns(
            PyfarDeprecationWarning,
            match="This function will be deprecated in pyfar 0.8.0 in favor "):
        coords.find_nearest_sph(1, 1, 1, 1)

    if version.parse(pf.__version__) >= version.parse('0.8.0'):
        with pytest.raises(TypeError):
            coords.find_nearest_sph(1, 1, 1, 1)


def test_deprecations_freq_range_parameter_warnings():
    sweep = pfs.exponential_sweep_time(256, (100, 10000))

    with pytest.warns(
            PyfarDeprecationWarning,
            match="freq_range parameter will be deprecated in pyfar 0.8.0 in "
            "favor of frequency_range"):
        pf.dsp.deconvolve(sweep, sweep, 256, freq_range=(20, 20e3))

    if version.parse(pf.__version__) >= version.parse('0.8.0'):
        with pytest.raises(TypeError):
            pf.dsp.deconvolve(sweep, sweep, 256, freq_range=(20, 20e3))

    with pytest.warns(
            PyfarDeprecationWarning,
            match="freq_range parameter will be deprecated in pyfar 0.8.0 in "
            "favor of frequency_range"):
        pf.dsp.regularized_spectrum_inversion(sweep, freq_range=(20, 20e3))

    if version.parse(pf.__version__) >= version.parse('0.8.0'):
        with pytest.raises(TypeError):
            pf.dsp.regularized_spectrum_inversion(sweep, freq_range=(20, 20e3))

    with pytest.warns(
            PyfarDeprecationWarning,
            match="freq_range parameter will be deprecated in pyfar 0.8.0 in "
            "favor of frequency_range"):
        gt = pf.dsp.filter.GammatoneBands(freq_range=(20, 20e3))

    if version.parse(pf.__version__) >= version.parse('0.8.0'):
        with pytest.raises(TypeError):
            gt = pf.dsp.filter.GammatoneBands(freq_range=(20, 20e3))

    with pytest.warns(
            PyfarDeprecationWarning,
            match="freq_range parameter will be deprecated in pyfar 0.8.0 in "
            "favor of frequency_range"):
        gt.freq_range

    if version.parse(pf.__version__) >= version.parse('0.8.0'):
        with pytest.raises(TypeError):
            gt.freq_range

    with pytest.warns(
            PyfarDeprecationWarning,
            match="freq_range parameter will be deprecated in pyfar 0.8.0 in "
            "favor of frequency_range"):
        pf.dsp.filter.erb_frequencies(freq_range=(20, 20e3))

    if version.parse(pf.__version__) >= version.parse('0.8.0'):
        with pytest.raises(TypeError):
            pf.dsp.filter.erb_frequencies(freq_range=(20, 20e3))

    with pytest.warns(
            PyfarDeprecationWarning,
            match="freq_range parameter will be deprecated in pyfar 0.8.0 in "
            "favor of frequency_range"):
        pf.dsp.filter.fractional_octave_bands(sweep, 8, freq_range=(20, 20e3))

    if version.parse(pf.__version__) >= version.parse('0.8.0'):
        with pytest.raises(TypeError):
            pf.dsp.filter.fractional_octave_bands(sweep, 8,
                                                  freq_range=(20, 20e3))


def test_deprecations_freq_range_parameter_renaming_results():
    sweep = pfs.exponential_sweep_time(256, (100, 10000))

    np.testing.assert_allclose(
        pf.dsp.deconvolve(sweep, sweep, 256, freq_range=(20, 20e3)).time,
        pf.dsp.deconvolve(sweep, sweep, 256, frequency_range=(20, 20e3)).time,
        rtol=0)

    np.testing.assert_allclose(
        pf.dsp.regularized_spectrum_inversion(sweep,
                                              freq_range=(20, 20e3)).time,
        pf.dsp.regularized_spectrum_inversion(sweep,
                                              frequency_range=(20, 20e3)).time,
        rtol=0)

    np.testing.assert_allclose(
        pf.dsp.filter.GammatoneBands(freq_range=(20, 20e3)).frequencies,
        pf.dsp.filter.GammatoneBands(frequency_range=(20, 20e3)).frequencies,
        rtol=0)

    np.testing.assert_allclose(
        pf.dsp.filter.erb_frequencies(freq_range=(20, 20e3)),
        pf.dsp.filter.erb_frequencies(frequency_range=(20, 20e3)),
        rtol=0)

    np.testing.assert_allclose(
        pf.dsp.filter.fractional_octave_bands(sweep, 8,
                                              freq_range=(20, 20e3)).time,
        pf.dsp.filter.fractional_octave_bands(sweep, 8,
                                              frequency_range=(20, 20e3)).time,
        rtol=0)

import numpy as np
import numpy.testing as npt     # noqa
from pytest import raises
import haiopy.haiopy as signal


def test_import_haiopy():
    try:
        import haiopy           # noqa
    except ImportError:
        assert False


def test_assert_match_for_math_operation():
    s = signal.Signal([1, 2, 3, 4], 44100)
    s1 = signal.Signal([1, 2, 3, 4], 48000)
    s2 = signal.Signal([1, 2, 3], 44100)

    # check with two signals
    signal._assert_match_for_math_operation((s, s), 'time')
    # check with one signal and one array like
    signal._assert_match_for_math_operation((s, [1, 2]), 'time')
    # check with more than two inputs
    signal._assert_match_for_math_operation((s, s, s), 'time')

    # check with only one argument
    with raises(TypeError):
        signal._assert_match_for_math_operation((s, s))
    # check with single input
    with raises(ValueError):
        signal._assert_match_for_math_operation(s, 'time')
    # check with invalid data type
    with raises(ValueError):
        signal._assert_match_for_math_operation((s, ['str', 'ing']), 'time')
    # check with complex data and time domain signal
    with raises(ValueError):
        signal._assert_match_for_math_operation(
            (s, np.array([1 + 1j])), 'time')
    # test signals with different sampling rates
    with raises(ValueError):
        signal._assert_match_for_math_operation((s, s1), 'time')
    # test signals with different n_samples
    with raises(ValueError):
        signal._assert_match_for_math_operation((s, s2), 'time')

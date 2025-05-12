from pyfar._utils import to_broadcastable_array
import numpy as np
import numpy.testing as npt
import pytest
import re

@pytest.mark.parametrize('dtype', [(int), (float)])
@pytest.mark.parametrize(('broadcast', 'a', 'b', 'a_target', 'b_target'), [
    (False, [1], [1, 2], [1], [1, 2]),
    (True,  [1], [1, 2], [1, 1], [1, 2])])
def test_to_broadcastable_array(dtype, broadcast, a, b, a_target, b_target):
    """Test function with different values for dtype and broadcast."""

    a_target = np.array(a_target, dtype=dtype)
    b_target = np.array(b_target, dtype=dtype)

    a_actual, b_actual = to_broadcastable_array(dtype, broadcast, a, b)

    npt.assert_equal(a_actual, a_target)
    npt.assert_equal(b_actual, b_target)
    assert isinstance(a_actual, type(a_target))
    assert isinstance(b_actual, type(b_target))


@pytest.mark.parametrize('broadcast', [(False), (True)])
def test_to_broadcastable_array_error(broadcast):
    """Test raising the ValueError in case input cannot be broadcasted."""

    a = [1, 2, 3]
    b = [1, 2]

    match = re.escape('Input parameters are of shape (3,), (2,)')

    with pytest.raises(ValueError, match=match):
        to_broadcastable_array(int, broadcast, a, b)

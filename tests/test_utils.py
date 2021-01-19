import pyfar
import numpy as np
import numpy.testing as npt


def test_copy():
    """ Test copy method used by several classes."""
    obj_list = [pyfar.Signal(1000, 44100),
                pyfar.Orientations(),
                pyfar.Coordinates(),
                pyfar.dsp.Filter()]

    for obj in obj_list:
        # Create copy
        obj_copy = obj.copy()
        # Check class
        assert isinstance(obj_copy, obj.__class__)
        # Check ID
        assert id(obj) != id(obj_copy)
        # Check attributes
        assert len(obj.__dict__) == len(obj_copy.__dict__)
        for key, value in obj.__dict__.items():
            assert key in obj_copy.__dict__.keys()
            value_copy = obj_copy.__dict__[key]
            if isinstance(value, np.ndarray):
                npt.assert_allclose(value, value_copy)
            else:
                assert value == value_copy

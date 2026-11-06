import pytest
import pyfar
from pyfar.classes.warnings import PyfarDeprecationWarning


def test_copy(
        time_data, frequency_data,
        filterFIR, filterIIR, filterSOS):
    """Test copy method used by several classes."""
    with pytest.warns(PyfarDeprecationWarning, match="will be deprecated"):
        orientations = pyfar.Orientations()

    obj_list = [pyfar.Signal(1000, 44100),
                orientations,
                pyfar.Coordinates(),
                filterFIR,
                filterIIR,
                filterSOS,
                time_data,
                frequency_data]

    for obj in obj_list:
        # Create copy
        if isinstance(obj, pyfar.Orientations):
            with pytest.warns(
                    PyfarDeprecationWarning, match="will be deprecated"):
                obj_copy = obj.copy()
        else:
            obj_copy = obj.copy()
        # Check class
        assert isinstance(obj_copy, obj.__class__)
        # Check ID
        assert id(obj) != id(obj_copy)
        # Check attributes
        assert len(obj.__dict__) == len(obj_copy.__dict__)
        # Check if copied objects are equal
        assert obj_copy == obj

import pyfar
import pytest
from pyfar.classes.warnings import PyfarDeprecationWarning

def test_copy(
        time_data, frequency_data,
        filterFIR, filterIIR, filterSOS):
    """ Test copy method used by several classes."""
    obj_list = [pyfar.Signal(1000, 44100),
                pyfar.Orientations(),
                pyfar.Coordinates(),
                # pyfar.classes.filter.Filter(),
                filterFIR,
                filterIIR,
                filterSOS,
                time_data,
                frequency_data]

    for obj in obj_list:
        # Create copy
        obj_copy = obj.copy()
        # Check class
        assert isinstance(obj_copy, obj.__class__)
        # Check ID
        assert id(obj) != id(obj_copy)
        # Check attributes
        assert len(obj.__dict__) == len(obj_copy.__dict__)
        # Check if copied objects are equal
        assert obj_copy == obj


@pytest.mark.filterwarnings('ignore::Warning')
def test_copy_sphericalvoronoi(
        sphericalvoronoi):
    """ Test copy method used by several classes."""
    # Create copy
    obj_copy = sphericalvoronoi.copy()
    # Check class
    assert isinstance(obj_copy, sphericalvoronoi.__class__)
    # Check ID
    assert id(sphericalvoronoi) != id(obj_copy)
    # Check attributes
    assert len(sphericalvoronoi.__dict__) == len(obj_copy.__dict__)
    # Check if copied objects are equal
    assert obj_copy == sphericalvoronoi

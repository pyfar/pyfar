import pyfar


def test_copy(sphericalvoronoi, time_data, frequency_data):
    """ Test copy method used by several classes."""
    obj_list = [pyfar.Signal(1000, 44100),
                pyfar.Orientations(),
                pyfar.Coordinates(),
                pyfar.classes.filter.Filter(),
                sphericalvoronoi,
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

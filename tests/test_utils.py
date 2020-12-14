import pyfar


def test_copy():
    obj_list = [pyfar.Signal(1000, 44100),
                pyfar.Orientations(),
                pyfar.Coordinates(),
                pyfar.dsp.Filter()]

    for obj in obj_list:
        # Create copy
        obj_copy = obj.copy()
        # Test for class
        assert isinstance(obj_copy, obj.__class__)
        # Test for location in memory
        assert id(obj) != id(obj_copy)
        # Test for properties
        for key, value in enumerate(obj_copy.__dict__)
        
    



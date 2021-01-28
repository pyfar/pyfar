"""
Provide methods used by several Classes.
"""
import numpy as np
from copy import deepcopy


def _eq___dict__(obj, other):
    """Check for equality of two objects."""
    if not isinstance(other, obj.__class__):
        return False
    for key, value in obj.__dict__.items():
        if isinstance(value, np.ndarray):
            if not np.array_equal(other.__dict__[key], value):
                return False
        elif other.__dict__[key] != value:
            return False
    return True


def copy(obj):
    """Return a deep copy of the object."""
    return deepcopy(obj)


def str_to_type(type_as_string, module='pyfar'):
    """
    Recursively find a PyfarType by passing in a valid type as a string.

    Parameters
    ----------
    type_as_string: string.
        A valid PyfarType.
    module: string.
        Either 'pyfar' or a submodule of pyfar, e.g. 'pyfar.spatial'
        The default is 'pyfar'.

    Returns
    ----------
    PyfarType: type.
        A valid PyfarType.
    """
    try:
        return getattr(sys.modules[module], type_as_string)
    except AttributeError:
        submodules = [attrib for attrib in dir(sys.modules[module])
            if not attrib.startswith('__') and attrib.islower()]
    except KeyError:
        return
    for submodule in submodules:
        PyfarType = str_to_type(type_as_string, module=f'{module}.{submodule}')
        if PyfarType:
            return PyfarType
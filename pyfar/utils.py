import numpy as np


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

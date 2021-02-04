"""
Provide methods used by several Classes.
"""
import sys
import numpy as np
from copy import deepcopy


def copy(obj):
    """Return a deep copy of the object."""
    return deepcopy(obj)
import numpy as np
from unittest import mock

from pyfar.coordinates import Coordinates


class MyOtherClass:
    def __init__(self):
        self.signal = np.sin(2 * np.pi * np.arange(0, 1, 1/10))

class NestedDataStruct:
    def __init__(self, n, comment, matrix, subobj, mylist, mydict):
        self._n = n
        self._comment = comment
        self._matrix = matrix
        self._subobj = subobj
        self._list = mylist
        self._dict = mydict
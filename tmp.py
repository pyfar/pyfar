import haiopy
import numpy as np
import matplotlib.pyplot as plt
import pprint

from haiopy import Signal
from haiopy import Coordinates

"""
Signal
"""
fs = 44100
t = np.arange(0, 5, 1/fs)
f1 = 1000
f2 = 4000
x = 0.5 + np.sin(2*np.pi * f1 * t) + 0.25 * np.sin(2*np.pi * f2 * t)
plt.plot(t[:1000], x[:1000])
# plt.show()

signal1 = Signal(x, fs)
signal1 = Signal(x, fs, None, 'time', 'energy', 'double')

"""
Coordinates
"""

coords = Coordinates()
coord1 = Coordinates(
    0.25,0.5,0.75,domain='cart', convention='right',
    unit='met', sh_order=8, comment="Let's check this out!")
coord1.show()

coords = Coordinates()
systems = coords._systems()

# pprint.pprint(systems)

my_dict = {
    'a': {
        'aa': {
            'aa0': 0,
            'aa1': 1},
        'ab': {
            'ab0': 0,
            'ab1': 1,
            'ab2': 2}}}

for l in my_dict:
    for dl in my_dict[l]:
        for dln in my_dict[l][dl]:
            print(my_dict[l][dl][dln])

from haiopy.coordinates import cart2sph

pt = [1, 1, 1]
# ptSph = cart2sph(pt[0], pt[1], pt[2])

print(coord1.get_sph())

signal1.position = Coordinates(1, 1, 1)
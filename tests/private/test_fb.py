#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 19:46:11 2020

@author: fabian
"""

import numpy as np
import haiopy
from haiopy import Coordinates


# %% test rotation

c = Coordinates([-1, 1], 0, 0)
c.show(np.array([0,1],dtype=bool))
c.rotate('Z', 90)


# %% slice and plot testing

# random spherical grid
az = np.random.rand(3,10,30) * 360
el = np.random.rand(3,10,30) * 180 - 90
cs = Coordinates(az, el, 1, 'sph', 'top_elev', 'deg')

mask = cs.get_slice('azimuth', 'deg', 0, 45, True)

# random cartesian grid
x  = np.random.rand(3,10,30) * 2 - 1
y  = np.random.rand(3,10,30) * 2 - 1
z  = np.random.rand(3,10,30) * 2 - 1
cc = Coordinates(x, y, z)

mask = cc.get_slice('x', 'met', .5, .25, True)

# random cylindircal grid
az = np.random.rand(3,10,30) * 360
z  = np.random.rand(3,10,30) * 2 - 1
cz = Coordinates(az, z, 1, 'cyl', 'top', 'deg')

mask = cz.get_slice('azimuth', 'deg', 90, 5, True)

# %% nearest and plot testing
# make random spherical grid
az = np.random.rand(3,10,30) * 360
el = np.random.rand(3,10,30) * 180 - 90
# horizontal plane
# az = np.linspace(0, 355, 72)
# el = np.zeros(72)

c = Coordinates(az, el, 1, 'sph', 'top_elev', 'deg')
# ax = haiopy.plot.scatter(c)

d, idx, mask = c.get_nearest_k(0,0,1, 50, 'sph', 'top_elev', 'deg', show=True)
# idx, mask = c.get_nearest_cart(0,0,1, 1, 'sph', 'top_elev', 'deg', show=True)
# idx, mask = c.get_nearest_sph(0,0,1, 90, 'sph', 'top_elev', 'deg', show=True)

# p1 = c.get_sph('top_elev', 'deg')[idx]
# p2 = c.get_sph('top_elev', 'deg')[~idx]


# %% General -----------------------------------------------------------------
# TODO: What is the idea of Audio()?
# TODO: How do we handle irregular data, e.g., ctave values?

# %% Signal ------------------------------------------------------------------

# TODO: Slack channel

# TODO: do not ifft immediately. 'domain' will become a property and self._data
#       will hold the data in the current domain. FFT/IFFT is done, when the
#       user changes 'domain' or pulls self.time if domain='freq' and vice
#       versa.

# TODO: The frequency domain constructor needs n_samples or isEven as an
#       additional argument.

# TODO: renaming
#       - self.signal_length -> self.length
#       - self.n_samples -> self.num_samples
#       - self._nbins -> self.num_bins (or even better self.num_freqs)
#       - self.frequencies becomes self.freqs
#       - self.position should be self.coordinates

# TODO: adding
#       - cdim -> self._data.ndim-1 (dimensions of the channels in self)
#       - cshape -> self._data.shape[:-1] (shape of channels in self)
#       - comment field for any information that might be helpful to understand
#         data in self
#       - signal() should be able to contain N coordinate classes (with custom
#         names?)

# TODO: slicing yields new signal object

# TODO: save / read functions for python and matlab after initial class
#       structure is ready

# TODO: Bugs
#       - constructor not working with list (x = haiopy.Signal([1,2,3,3],44100))
#       - __repr__ only works for 2D arrays

# TODO: Documentation
#       - add information about effect of self.signal_type

# generate audio signal
x = haiopy.Signal(np.array([1,2,3,4]),44100)

# TODO: should also work with list - conversion inside Signal for ease of use
x = haiopy.Signal([1,2,3,3],44100)

x.time = np.array([2,0,0,0])

x.freq = np.array([1,1,1])


# TODO: Use only length?
# TODO: Use sring argument to return length in seconds, samples, frequencies?
x.signal_length

x.n_samples
x.n_bins

# TODO: String argument to get times and frequencies in samples, s, ms, , kHz, etc?
x.times
x.frequencies
# TODO: get times in samples?
x.samples

# TODO: position
#       - Rename (see above)
#       - a signal() should be able to have between 0 and cdim coordinates()
#       - How are different coordinates distinguisehd? They coould explicitly
#         assigned to an axis. Do lists of objects work in python? Would that
#         be a good idea? We could also implictly assign coordinates() to an
#         axis by matchting the length and throw an error if to axis have the
#         same dimension.
x.position = haiopy.Coordinates([0,1], [1,0], [0,0])

# TODO: orientation
#       - if position gets assigned to an axis, orientation also need to


# TODO: Only shows two dimensions so far...
x

# TODO: What is depending on signal_type
x.signal_type


# %% Orientation -------------------------------------------------------------

# URGH: not discussed yet - first thoughts fb
# Concept: Orientation() could be a propertiy of Coordinates(). Alternatively,
#          both could be properties of the object that holds them. The latter
#          is more flexible and is the current implementation.

# Should this be part of the coordinates module?

# TODO: add Quaternion, Yaw/Pitch/Roll - agreed

# %% Position ---------------------------------------------------------------

# DONE: A position is a Coordinates object with 1 point. For simplicity we
#       do not want a separate class for that.
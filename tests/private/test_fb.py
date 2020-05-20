#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 19:46:11 2020

@author: fabian
"""

import numpy as np
import haiopy
from haiopy import Coordinates

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

# %% Coordinates -------------------------------------------------------------

# CONCEPT: My understanding is that Coordinates() are local coordinates for
#          object that they are asigned to. Orientation() and Position() would
#          relate the local coordinates to world coordinates. Is this the way
#          it was intended?

# TODO: In analogy to signal(), store data in the last used coordinate system
#       and only convert upon request

# TODO: Clear naming of coordinate systems and parameters of the coordinate
#       systems. E.g. 'spherical 1' and 'azimuth', 'colatitude', and 'radius'

# TODO: find way to return coordinates in different units 'degree', 'radians',
#       'meter', 'millimeter'
# TODO: Degrees instead of radians, or string to get either (please, please, please)?

# TODO: Document how new coordinate conventions can be added

# TODO: add plot function

# TODO: add comment field, e.g. for name of the sampling scheme
# TODO: can contain sampling weights?
# TODO: can have a spherical harmonics maximum order (read/write)?


# TODO: make all constructors work with scalars and array_like?
c = haiopy.Coordinates.from_spherical(1, np.pi/2, 0)

c = haiopy.Coordinates([0,1], [1,0], [0,0])

# TODO: short names 'cart', 'sph', 'az', etc...?
c.cartesian


# TODO: find_nearest_point
# - Make it work with numbers/arrays as input (conversion to class inside function)
# - Make it work for different coordinate systems as input (additional string parameter)
# - Make it work with arrays as input
# - Features from AKsubGrid

# TODO: renaming and additional functionality (?)
# - self.nearest_point: instead of find_nearest_point to make it shorter
# - self.nearest_points: as nearest point, but return everything within a given
#   range
# - self.slice: return slices, e.g., horizontal plane within given tolerance
#   (see AKsubGrid from AKtools)
d, i = c.find_nearest_point(haiopy.Coordinates([0], [.7], [0]))


# %% tmp testing

c = Coordinates( 1, 0, 0)
# c = Coordinates( 0, 1, 0)
# c = Coordinates(-1, 0, 0)
# c = Coordinates( 0,-1, 0)
# c = Coordinates( 0, 0, 1)
# c = Coordinates( 0, 0,-1)
print(c.get_sph('front', 'deg'))


# %% Orientation -------------------------------------------------------------

# URGH: not discussed yet - first thoughts fb
# Concept: Orientation() could be a propertiy of Coordinates(). Alternatively,
#          both could be properties of the object that holds them. The latter
#          is more flexible and is the current implementation.

# Should this be part of the coordinates module?

# TODO: add Quaternion, Yaw/Pitch/Roll - agreed

# %% Position ---------------------------------------------------------------

# TODO: Do we want a position? I think yes - for completeness.
"""
Collection of sampling schemes and related functionality. For information on
the used coordinate systems refer to the
:py:mod:`Coordinates documentation <pyfar.classes.coordinates>`.

.. warning::

    This module will be deprecated in pyfar v0.8.0 in favor of
    :py:mod:`spharpy.samplings <spharpy.samplings>`.
"""

from .spatial import SphericalVoronoi, calculate_sph_voronoi_weights
from .samplings import (
    cart_equidistant_cube, sph_dodecahedron, sph_icosahedron, sph_equiangular,
    sph_gaussian, sph_extremal, sph_t_design, sph_equal_angle,
    sph_great_circle, sph_lebedev, sph_fliege, sph_equal_area)


__all__ = [
    'SphericalVoronoi',
    'calculate_sph_voronoi_weights',
    'samplings',
    'cart_equidistant_cube',
    'sph_dodecahedron',
    'sph_icosahedron',
    'sph_equiangular',
    'sph_gaussian',
    'sph_extremal',
    'sph_t_design',
    'sph_equal_angle',
    'sph_great_circle',
    'sph_lebedev',
    'sph_fliege',
    'sph_equal_area']

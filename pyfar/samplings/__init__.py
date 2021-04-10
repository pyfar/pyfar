"""
Collection of sampling schemes and related functionality.
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

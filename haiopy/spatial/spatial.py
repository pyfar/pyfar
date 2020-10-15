import numpy as np
from scipy import spatial as spat


class SphericalVoronoi(spat.SphericalVoronoi):

    def __init__(self, sampling, round_decimals=12, center=0.0):
        """Calculate a Voronoi diagram on the sphere for the given samplings
        points.

        Parameters
        ----------
        sampling : SamplingSphere
            Spherical sampling points in Carthesian coordinates
        round_decimals : int
            Number of decimals to be rounded for checking for equal radius.
            The default is 12.
        center : double
            Center point of the voronoi diagram. The default is 0.

        Returns
        -------
        voronoi : SphericalVoronoi
            Spherical voronoi diagram as implemented in scipy.

        """
        points = sampling.get_cart()
        radius = sampling.get_sph()[:, -1]
        radius_round = np.unique(np.round(radius, decimals=round_decimals))
        if len(radius_round) > 1:
            raise ValueError("All sampling points need to be on the \
                    same radius.")
        super().__init__(points, radius_round, center)


def calculate_sampling_weights_with_spherical_voronoi(
        sampling, round_decimals=12):
    """Calculate the sampling weights for numeric integration.

    Parameters
    ----------
    sampling : Coordinates
        Sampling points on a sphere, i.e., all points must have the same
        radius.
    round_decimals : int, optional
        Round to decimals to check for equal radius and duplicate points in the
        Voronoi diagram. The default is 12.

    Returns
    -------
    weigths : ndarray, np.double
        Sampling weights

    """
    # get Voronoi diagram
    sv = SphericalVoronoi(sampling, round_decimals=round_decimals)

    # prepare points for calculating the are
    # sort vertives of each region to be in clockwise order
    sv.sort_vertices_of_regions()
    # detect duplicate vertices
    unique_verts, idx_uni = np.unique(
        np.round(sv.vertices, decimals=10),
        axis=0,
        return_index=True)

    # search tree for efficient calculation of the areas
    searchtree = spat.cKDTree(unique_verts)
    area = np.zeros(sampling.csize, np.double)

    # calculate and normalize the area
    for idx, region in enumerate(sv.regions):
        _, idx_nearest = searchtree.query(sv.vertices[np.array(region)])
        mask_unique = np.sort(np.unique(idx_nearest, return_index=True)[1])
        mask_new = idx_uni[idx_nearest[mask_unique]]

        area[idx] = _poly_surface_area(sv.vertices[mask_new])

    area = area / np.sum(area)

    return area


def _unit_normal(a, b, c):
    """Calculate the normal vector for a polygon defined by the points a, b,
    and c.
    """
    x = np.linalg.det(
        [[1, a[1], a[2]],
         [1, b[1], b[2]],
         [1, c[1], c[2]]])
    y = np.linalg.det(
        [[a[0], 1, a[2]],
         [b[0], 1, b[2]],
         [c[0], 1, c[2]]])
    z = np.linalg.det(
        [[a[0], a[1], 1],
         [b[0], b[1], 1],
         [c[0], c[1], 1]])

    norm = np.sqrt(x**2 + y**2 + z**2)

    return (x/norm, y/norm, z/norm)


def _poly_surface_area(polygon):
    """Calculate the surface area of a polygon.
    """
    if len(polygon) < 3:
        return 0
    total = [0.0, 0.0, 0.0]
    N = len(polygon)
    for i in range(N):
        vi1 = polygon[i]
        vi2 = polygon[np.mod((i+1), N)]
        prod = np.cross(vi1, vi2)
        total[0] += prod[0]
        total[1] += prod[1]
        total[2] += prod[2]
    result = np.dot(total, _unit_normal(polygon[0], polygon[1], polygon[2]))
    return np.abs(result/2)

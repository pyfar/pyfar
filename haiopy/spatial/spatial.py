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
        sampling, normalize=True, center=[0, 0, 0], round_decimals=12):
    """Calculate the sampling weights for numeric integration.

    This is wrapper for scipy.spatial.SphericalVoronoi and uses the class
    method calculate_areas() to calculate the weights.

    Parameters
    ----------
    sampling : Coordinates
        Sampling points on a sphere, i.e., all points must have the same
        radius.
    normalize : boolean, optional
        Normalize the samplings weights to sum(weights)=1. Otherwise the
        weights sum to :math:`4 \\pi r^2`. The default is True.
    center : list
        Center of the spherical sampling grid. The default is [0, 0, 0].
    round_decimals : int, optional
        Round to `round_decimals' digits to check for equal radius. The
        default is 12.

    Returns
    -------
    weigths : ndarray, np.double
        Sampling weights of size samplings.csize.

    """
    # get Voronoi diagram
    sv = SphericalVoronoi(sampling, round_decimals, center)

    # get the area
    weights = sv.calculate_areas()

    if normalize:
        weights /= np.sum(weights)

    return weights

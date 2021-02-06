import deepdiff
import numpy as np
from pyfar import utils
from scipy import spatial as spat
from pyfar import Coordinates


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

    def copy(self):
        """Return a deep copy of the Coordinates object."""
        return utils.copy(self)

    def _encode(self):
        """Return dictionary for the encoding."""
        return self.copy().__dict__

    @classmethod
    def _decode(cls, obj_dict):
        """Decode object based on its respective `_encode` counterpart."""
        sampling = Coordinates(
        obj_dict['points'][:, 0],
        obj_dict['points'][:, 1],
        obj_dict['points'][:, 2],
        domain='cart')
        return cls(sampling, center=obj_dict['center'])

    def __eq__(self, other):
        """Check for equality of two objects."""
        return not deepdiff.DeepDiff(
            self, other, ignore_type_in_groups=[
                (np.int32, np.intc), (np.int64, np.intc)])


def calculate_sph_voronoi_weights(
        sampling, normalize=True, center=[0, 0, 0], round_decimals=12):
    """Calculate sampling weights for numeric integration.

    This is wrapper for scipy.spatial.SphericalVoronoi and uses the class
    method calculate_areas() to calculate the weights. It requires a spherical
    sampling grid with a single radius.

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
        Round to `round_decimals` digits to check for equal radius. The
        default is 12.

    Returns
    -------
    weigths : ndarray, np.double
        Sampling weights of size samplings.csize.

    """
    # get Voronoi diagram
    if sampling.csize <= 3:
        raise ValueError(
            'The number of points needs to be at least 4',
            'to generate a valid SphericalVoronoi diagram.')
    sv = SphericalVoronoi(sampling, round_decimals, center)

    # get the area
    weights = sv.calculate_areas()

    if normalize:
        weights /= np.sum(weights)

    return weights

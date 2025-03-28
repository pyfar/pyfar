"""This module provides functions to calculate spherical Voronoi diagrams."""
import deepdiff
import numpy as np
from scipy import spatial as spat
from pyfar import Coordinates
from copy import deepcopy
import warnings
from pyfar.classes.warnings import PyfarDeprecationWarning


class SphericalVoronoi(spat.SphericalVoronoi):
    """
    Voronoi diagrams on the surface of a sphere. Note that
    :py:func:`calculate_sph_voronoi_weights` can be used directly, if only the
    sampling weights are needed.
    """

    def __init__(self, sampling, round_decimals=12, center=0.0):
        """
        Calculate a Voronoi diagram on the sphere for the given samplings
        points.

        This function will be deprecated in pyfar 0.8.0 in favor
        of :py:func:`spharpy.samplings.spherical_voronoi`.

        Parameters
        ----------
        sampling : Coordinates
            Spherical sampling.
        round_decimals : int
            Number of decimals to be rounded for checking for equal radius.
            The default is ``12``.
        center : double
            Center point of the voronoi diagram. The default is ``0``.

        Returns
        -------
        voronoi : SphericalVoronoi
            Spherical voronoi diagram as implemented in
            :py:mod:`scipy.spatial`.

        See Also
        --------
        :py:func:`calculate_sph_voronoi_weights`

        """
        warnings.warn((
            "This function will be deprecated in pyfar 0.8.0 in favor "
            "of spharpy.samplings.spherical_voronoi."),
                PyfarDeprecationWarning, stacklevel=2)

        points = sampling.cartesian
        radius = sampling.radius
        radius_round = np.unique(np.round(radius, decimals=round_decimals))
        if len(radius_round) > 1:
            raise ValueError("All sampling points need to be on the \
                    same radius.")
        super().__init__(points, radius_round[0], center)

    def copy(self):
        """Return a copy of the Voronoi object."""
        return deepcopy(self)

    def _encode(self):
        """Return object in a proper encoding format."""
        # Use public interface of the scipy super-class to prevent
        # error in case of chaning super-class implementations
        return {'points': self.points, 'center': self.center}

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
        return not deepdiff.DeepDiff(self, other)


def calculate_sph_voronoi_weights(
        sampling, normalize=True, center=[0, 0, 0], round_decimals=12):
    r"""
    Calculate sampling weights for numeric integration.

    This function will be deprecated in pyfar 0.8.0 in favor
    of :py:func:`spharpy.samplings.calculate_sampling_weights`.

    Uses the class method
    :py:meth:`~scipy.spatial.SphericalVoronoi.calculate_areas`
    from :py:class:`SphericalVoronoi`
    to calculate the weights. It requires a spherical sampling grid with a
    single radius and uses :py:class:`scipy.spatial.SphericalVoronoi` in the
    background.

    Parameters
    ----------
    sampling : Coordinates
        Sampling points on a sphere, i.e., all points must have the same
        radius.
    normalize : boolean, optional
        Normalize the samplings weights to ``sum(weights)=1``. Otherwise the
        weights sum to :math:`4 \pi r^2`. The default is ``True``.
    center : list
        Center of the spherical sampling grid. The default is ``[0, 0, 0]``.
    round_decimals : int, optional
        Round to `round_decimals` digits to check for equal radius. The
        default is ``12``.

    Returns
    -------
    weigths : ndarray, double
        Sampling weights of size `samplings.csize`.

    """
    warnings.warn((
        "This function will be deprecated in pyfar 0.8.0 in favor "
        "of spharpy.samplings.calculate_sampling_weights."),
            PyfarDeprecationWarning, stacklevel=2)

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

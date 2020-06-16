"""
Plot for spatially distributed data.
"""
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from haiopy.coordinates import Coordinates


def scatter(coordinates, projection='3d', ax=None):
    """Plot the x, y, and z coordinates as a point cloud in three-dimensional
    space.

    Parameters
    ----------
    coordinates : Coordinates
        Coordinates object with respective positions.
    projection : '3d', 'ortho'
        Projection to be used for the plot. Only three-dimensional projections
        are supported.
    ax : matplotlib.axis (optional)
        If no axis is defined, the currently active axis is selected through
        matplotlib or a new axis in a new figure is created.

    Returns
    ax : matplotlib.axes
        The axis used for the plot.

    """
    if not isinstance(coordinates, Coordinates):
        raise ValueError("The coordinates need to be a Coordinates object")

    if ax is None:
        ax = plt.gca(projection=projection)

    if not 'Axes3D' in ax.__str__():
        raise ValueError("Only three-dimensional axes supported.")

    ax.scatter(
        coordinates.get_cart()[..., 0],
        coordinates.get_cart()[..., 1],
        coordinates.get_cart()[..., 2])

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')

    return ax

"""
Plot for spatially distributed data.
"""
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import copy

from haiopy.coordinates import Coordinates


def scatter(coordinates, projection='3d', ax=None, set_ax=True, **kwargs):
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
        If no axis is defined, a new axis in a new figure is created.
    set_ax: boolean
        Set the limits of the axis according to the points in coordinates. The
        default is True.
    **kwargs :
        additional key value arguments are passed to matplotlib.pyplot.scatter.

    Returns
    ax : matplotlib.axes
        The axis used for the plot.

    """
    if not isinstance(coordinates, Coordinates):
        raise ValueError("The coordinates need to be a Coordinates object")

    if ax is None:
        # create equal aspect figure for distortion free display
        # (workaround for ax.set_aspect('equal', 'box'), which is currently not
        #  working for 3D axes.)
        plt.figure(figsize=plt.figaspect(1.))
        ax = plt.gca(projection=projection)

    if not 'Axes3D' in ax.__str__():
        raise ValueError("Only three-dimensional axes supported.")

    # add defaults to kwargs
    if not 'marker' in kwargs:
        kwargs['marker'] = '.'
    if not 'c' in kwargs:
        kwargs['c'] = 'k'

    # copy to avoid changing the coordinate system of the original object
    c   = copy.deepcopy(coordinates)
    xyz = c.get_cart()

    # plot
    ax.scatter(
        xyz[..., 0],
        xyz[..., 1],
        xyz[..., 2], **kwargs)

    # labeling
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')

    # equal axis limits for distortion free  display
    if set_ax:
        # unfortunately ax.set_aspect('equal') does not work on Axes3D
        ax_lims = (np.min(xyz)-.15*np.abs(np.min(xyz)),
                   np.max(xyz)+.15*np.abs(np.max(xyz)))

        ax.set_xlim(ax_lims)
        ax.set_ylim(ax_lims)
        ax.set_zlim(ax_lims)

    return ax

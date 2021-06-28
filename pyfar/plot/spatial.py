"""
Plot for spatially distributed data.
"""
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pyfar as pf

__all__ = [Axes3D]


def scatter(points, projection='3d', ax=None, set_ax=True, **kwargs):
    """Plot the x, y, and z coordinates as a point cloud in three-dimensional
    space.

    Parameters
    ----------
    points : array_like, shape (N, 3) or (3,)
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
        Additional key value arguments are passed to matplotlib.pyplot.scatter.

    Returns
    ax : matplotlib.axes
        The axis used for the plot.

    """
    points = np.atleast_2d(points).astype(np.float64)

    # default marker size
    kwargs['s'] = kwargs.get('s', np.clip(8e3 / points.shape[0], 4, 100))

    # plot with plotstyle
    with pf.plot.context():
        ax = _setup_axes(projection, ax, set_ax,
                         bounds=(np.min(points), np.max(points)), **kwargs)

        ax.scatter(points[..., 0], points[..., 1], points[..., 2], **kwargs)

    return ax


def quiver(
        origins, endpoints, projection='3d', ax=None, set_ax=True, **kwargs):
    """Plot vectors from their origins (x, y, z) to their endpoints (u, v, w).

    Parameters
    ----------
    origins : array_like, shape (N, 3) or (3,)
        The coordinates of the origins of the vectors.
    endpoints : array_like, shape (N, 3) or (3,)
        The coordinates of the endpoints of the vectors.
    projection : '3d', 'ortho'
        Projection to be used for the plot. Only three-dimensional projections
        are supported.
    ax : matplotlib.axis (optional)
        If no axis is defined, a new axis in a new figure is created.
    set_ax: boolean
        Set the limits of the axis according to the points in coordinates. The
        default is True.
    **kwargs :
        Additional key value arguments are passed to matplotlib.pyplot.quiver.

    Returns
    ax : matplotlib.axes
        The axis used for the plot.

    """
    origins = np.atleast_2d(origins).astype(np.float64)
    endpoints = np.atleast_2d(endpoints).astype(np.float64)

    min_val = min(np.min(origins), np.min(endpoints))
    max_val = max(np.max(origins), np.max(endpoints))

    # plot with plotstyle
    with pf.plot.context():
        ax = _setup_axes(
            projection, ax, set_ax, bounds=(min_val, max_val), **kwargs)

        ax.quiver(*origins.T, *endpoints.T, **kwargs)

    return ax


def _setup_axes(projection=Axes3D.name, ax=None,
                set_ax=True, bounds=(-1, 1), **kwargs):
    """Setup axes' limits and labels for 3D-plots.

    Parameters
    ----------
    projection : '3d', 'ortho'
        Projection to be used for the plot. Only three-dimensional projections
        are supported.
    ax : matplotlib.axis (optional)
        If no axis is defined, a new axis in a new figure is created.
    set_ax: boolean
        Set the limits of the axis according to the points in coordinates. The
        default is True.
    bounds: tuple (min, max)
        The lower and upper boundaries of the data to be plotted. This is used
        for the axes' limits. Default is Axes3D.name ('3d').
    **kwargs :
        Additional key value arguments are passed to matplotlib.pyplot.scatter.

    Returns
    ax : matplotlib.axes
        The axis used for the plot.

    """
    if ax is None:
        # create equal aspect figure for distortion free display
        # (workaround for ax.set_aspect('equal', 'box'), which is currently not
        #  working for 3D axes.)
        plt.figure(figsize=plt.figaspect(1.))
        ax = plt.subplot(111, projection=projection)

    if 'Axes3D' not in ax.__str__():
        raise ValueError("Only three-dimensional axes supported.")

    # add defaults to kwargs
    kwargs['marker'] = kwargs.get('marker', '.')

    # labeling
    ax.set_xlabel('x in m')
    ax.set_ylabel('y in m')
    ax.set_zlabel('z in m')

    # equal axis limits for distortion free  display
    if set_ax:
        # unfortunately ax.set_aspect('equal') does not work on Axes3D
        ax_lims = [bounds[0]-.15*np.abs(bounds[0]),
                   bounds[1]+.15*np.abs(bounds[1])]
        if not ax.get_autoscale_on():
            if ax_lims[0] > ax.get_xlim()[0]:
                ax_lims[0] = ax.get_xlim()[0]
            if ax_lims[1] < ax.get_xlim()[1]:
                ax_lims[1] = ax.get_xlim()[1]

        ax.set_xlim(ax_lims)
        ax.set_ylim(ax_lims)
        ax.set_zlim(ax_lims)

    return ax

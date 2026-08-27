from pyfar.classes.warnings import PyfarDeprecationWarning
import numpy as np
import warnings
import functools


# Decorator function to rename parameters to be deprecated
def rename_arg(arg_map, warning_message):
    """
    Function for deprecating or renaming arguments.

    Intercepts input if a deprecated argument is passed, replaces it with
    a new argument and raises a PyfarDeprecationWarning.

    Parameters
    ----------
    arg_map : dictionary
        Map with deprecated argument to intercept and new argument to
        replace it with: ``{"deprecated_argument": "new_argument"}``
    warning_message: string
        Message to be thrown with deprecation warning.

    Returns
    -------
    function : function
        Modified function with replaced arguments.

    Examples
    --------
    Following example shows how a deprecated argument can be replaced by a
    new argument while throwing a deprecation warning:

        >>> from pyfar._utils import rename_arg
        >>>
        >>> @rename_arg({"old_arg": "new_arg"}, "old-arg will be deprecated in"
        >>>             " version x.x.x in favor of new_arg")
        >>> def function(arg1, arg2, new_arg):
        >>>     return arg1, arg2, new_arg
        >>>
        >>> function(1, 2, old_arg=3)
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            new_kwargs = {}
            for kwarg, value in kwargs.items():
                if kwarg in arg_map:
                    warnings.warn(
                        (warning_message), PyfarDeprecationWarning,
                        stacklevel=2)
                new_kwargs[arg_map.get(kwarg, kwarg)] = value
            return func(*args, **new_kwargs)
        return wrapper
    return decorator


def to_broadcastable_array(dtype, broadcast, *args):
    """
    Cast all *args to numpy arrays and check if shapes can be broadcasted.

    Raises a value error if shapes can not be broadcasted

    Parameters
    ----------
    dtype : string
        The dtype for creating the numpy arrays, e.g., ``float``
    broadcast : Bool
        If ``True`` broadcasted arrays are returned. If ``False`` it is only
        checked if arrays can be broadcasted.
    *args : scalar, array like
        Scalar and array likes that are converted to numpy arrays

    Returns
    -------
    *args : list of numpy arrays
        The input *args as numpy arrays
    """
    arrays = [np.array(arg, dtype=dtype) for arg in args]
    shapes = [array.shape for array in arrays]

    try:
        if broadcast:
            arrays = np.broadcast_arrays(*arrays)
        else:
            np.broadcast_shapes(*shapes)
    except ValueError as e:
        shapes = [str(shape) for shape in shapes]
        raise ValueError(
            f"Input parameters are of shape {', '.join(shapes)} and cannot be "
            "be broadcasted to a common shape") from e

    return arrays

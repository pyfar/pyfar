from pyfar.classes.warnings import PyfarDeprecationWarning
import warnings
import functools


# Decorator function to rename parameters to be deprecated
def rename_arg(arg_map, warning_message):
    """
    Function for deprecating or renaming arguments.

    Intercepts input if a deprecated argument is passed, replaces it with
    a new argument and raises a PyfarDeprecationWarning.

    Parameters
    -----------
    arg_map : dictionary
        Map with deprecated argument to intercept and new argument to
        replace it with: ``{"deprecated_argument": "new_argument"}``
    warning_message: string
        Message to be thrown with deprecation warning.

    Returns
    ---------
    function : function
        Modified function with replaced arguments.

    Examples
    ---------
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
                    warnings.warn((warning_message), PyfarDeprecationWarning)
                new_kwargs[arg_map.get(kwarg, kwarg)] = value
            return func(*args, **new_kwargs)
        return wrapper
    return decorator

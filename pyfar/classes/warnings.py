"""
The following documents the pyfar warning classes.
"""


class PyfarDeprecationWarning(Warning):
    """
    This class creates a custom PyfarDeprecationWarning inheriting from
    the `Warning` class. It's supposed to be used instead of
    `DeprecationWarning` or `PendingDeprecationWarning`. Both Warnings are
    ignored by default, so using `warnings.simplefilter()` to show the warning
    messages can be avoided by using this class. Deprecated features need to be
    removed after two minor versions.

    Examples
    --------
    To use the warning it needs to be imported from pyfar and it can be raised
    by using warnings.warn().

    >>> from pyfar.classes.warnings import PyfarDeprecationWarning
    >>> warnings.warn('Deprecation Message', PyfarDeprecationWarning)

    It needs to be tested, if the warning gets raised, with:

    >>> import pytest
    >>> import pyfar as pf
    >>> from pyfar.classes.warnings import PyfarDeprecationWarning
    >>> with pytest.warns(PyfarDeprecationWarning,
    >>>                   match="Deprecation Message"):
    >>>    warnings.warn('Deprecation Message', PyfarDeprecationWarning)

    and also if the deprecated features got removed after two minor versions.
    The following shows the example of the the pyfar signal exponential_sweep()
    , which got deprecated in minor version '0.3.0' and removed in '0.5.0'.

    >>> if version.parse(pf.__version__) >= version.parse('0.5.0'):
    >>>     with pytest.raises(AttributeError):
    >>>         # remove exponential_sweep() from pyfar 0.5.0!
    >>>         pf.signals.exponential_sweep(2**10, [1e3, 20e3])

    """
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return repr(self.message)

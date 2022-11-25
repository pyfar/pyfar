class PyfarDeprecationWarning(Warning):
    """
    This class creates a custom PyfarDeprecationWarning inheriting from
    the Warning class.
    """
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return repr(self.message)

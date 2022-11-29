from pyfar.classes.warnings import PyfarDeprecationWarning
import pytest
import warnings


def test_warn_PyfarDeprecationWarning():
    # tests if PyfarDeprecationWarning will be raised.
    with pytest.warns(PyfarDeprecationWarning, match="Deprecation Message"):
        warnings.warn('Deprecation Message', PyfarDeprecationWarning)

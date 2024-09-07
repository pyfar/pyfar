"""
Test imports separately for each module and top level functions/classes. These
tests will give hints in case a single export breaks, whereas im `import pyfar`
would fail completely and make possible issues harder to find.
"""
import importlib


def test_import_importlib():
    print(importlib.util.find_spec('pyfar'))


def test_import_pyfar():
    import pyfar                             # noqa: F401


def test_import_classes():
    from pyfar import Signal                 # noqa: F401
    from pyfar import TimeData               # noqa: F401
    from pyfar import FrequencyData          # noqa: F401

    from pyfar import Coordinates            # noqa: F401
    from pyfar import Orientations           # noqa: F401

    from pyfar import FilterSOS              # noqa: F401
    from pyfar import FilterFIR              # noqa: F401
    from pyfar import FilterIIR              # noqa: F401


def test_import_submodules():
    from pyfar import dsp                    # noqa: F401
    from pyfar.dsp import fft                # noqa: F401
    from pyfar.dsp import filter             # noqa: F401
    from pyfar import io                     # noqa: F401
    from pyfar import plot                   # noqa: F401
    from pyfar import samplings              # noqa: F401
    from pyfar import signals                # noqa: F401
    from pyfar.signals import files          # noqa: F401


def test_import_functions():
    from pyfar import add                    # noqa: F401
    from pyfar import subtract               # noqa: F401
    from pyfar import multiply               # noqa: F401
    from pyfar import divide                 # noqa: F401
    from pyfar import power                  # noqa: F401
    from pyfar import matrix_multiplication  # noqa: F401

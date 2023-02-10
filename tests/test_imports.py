import importlib
from types import ModuleType
from inspect import isclass


def test_import_importlib():
    print(importlib.util.find_spec('pyfar'))


def test_import_pyfar():
    import pyfar
    assert isinstance(pyfar, ModuleType)


def test_import_classes():
    from pyfar import Signal
    from pyfar import TimeData
    from pyfar import FrequencyData

    from pyfar import Coordinates
    from pyfar import Orientations

    from pyfar import FilterSOS
    from pyfar import FilterFIR
    from pyfar import FilterIIR

    assert isclass(Signal)
    assert isclass(TimeData)
    assert isclass(FrequencyData)
    assert isclass(Coordinates)
    assert isclass(Orientations)
    assert isclass(FilterSOS)
    assert isclass(FilterFIR)
    assert isclass(FilterIIR)


def test_import_submodules():
    import pyfar
    assert isinstance(pyfar.dsp, ModuleType)
    assert isinstance(pyfar.dsp.fft, ModuleType)
    assert isinstance(pyfar.dsp.filter, ModuleType)
    assert isinstance(pyfar.io, ModuleType)
    assert isinstance(pyfar.samplings, ModuleType)
    assert isinstance(pyfar.plot, ModuleType)
    assert isinstance(pyfar.signals, ModuleType)
    assert isinstance(pyfar.signals.files, ModuleType)

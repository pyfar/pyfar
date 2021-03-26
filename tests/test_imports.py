import importlib


def test_import_importlib():
    print(importlib.util.find_spec('pyfar'))


def test_import_pyfar():
    import pyfar
    __all__ = [pyfar]
    return __all__


def test_import_classes():
    from pyfar import Signal
    from pyfar import TimeData
    from pyfar import FrequencyData

    from pyfar import Coordinates
    from pyfar import Orientations

    from pyfar.dsp import Filter
    from pyfar.dsp import FilterSOS
    from pyfar.dsp import FilterFIR
    from pyfar.dsp import FilterIIR

    __all__ = [
        Signal,
        TimeData,
        FrequencyData,
        Coordinates,
        Orientations,
        Filter,
        FilterSOS,
        FilterFIR,
        FilterIIR
    ]
    return __all__


def test_import_submodules():
    import pyfar
    assert pyfar.dsp
    assert pyfar.dsp.fft
    assert pyfar.io
    assert pyfar.spatial
    assert pyfar.plot

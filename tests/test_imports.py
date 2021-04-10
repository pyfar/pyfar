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

    from pyfar import FilterSOS
    from pyfar import FilterFIR
    from pyfar import FilterIIR

    __all__ = [
        Signal,
        TimeData,
        FrequencyData,
        Coordinates,
        Orientations,
        FilterSOS,
        FilterFIR,
        FilterIIR
    ]
    return __all__


def test_import_submodules():
    import pyfar
    assert pyfar.dsp
    assert pyfar.dsp.fft
    assert pyfar.dsp.filter
    assert pyfar.io
    assert pyfar.samplings
    assert pyfar.plot
    assert pyfar.signals

import scipy.io.wavfile as wavfile
import os.path
import warnings
import numpy as np
import sofa
import json
import zipfile
import copy
import io
import sys

from pyfar import Signal
from pyfar import Coordinates
import pyfar.dsp.classes as fo


def read_wav(filename):
    """
    Import a WAV file as signal object.

    This method is based on scipy.io.wavfile.read().

    Parameters
    ----------
    filename : string or open file handle
        Input wav file.

    Returns
    -------
    signal : signal instance
        An audio signal object from the pyfar Signal class
        containing the audio data from the WAV file.

    Notes
    -----
    * This function is based on scipy.io.wavfile.write().
    * This function cannot read wav files with 24-bit data.
    """
    sampling_rate, data = wavfile.read(filename)
    signal = Signal(data.T, sampling_rate, domain='time')
    return signal


def write_wav(signal, filename, overwrite=True):
    """
    Write a signal as a WAV file.

    Parameters
    ----------
    signal : Signal object
        An audio signal object from the pyfar Signal class.

    filename : string or open file handle
        Output wav file.

    overwrite : bool
        Select wether to overwrite the WAV file, if it already exists.
        The default is True.

    Notes
    -----
    * This function is based on scipy.io.wavfile.write().
    * Writes a simple uncompressed WAV file.
    * Signals of shape larger than 1D are flattened.
    * The bits-per-sample and PCM/float will be determined by the data-type.

    Common data types: [1]_

    =====================  ===========  ===========  =============
         WAV format            Min          Max       NumPy dtype
    =====================  ===========  ===========  =============
    32-bit floating-point  -1.0         +1.0         float32
    32-bit PCM             -2147483648  +2147483647  int32
    16-bit PCM             -32768       +32767       int16
    8-bit PCM              0            255          uint8
    =====================  ===========  ===========  =============

    Note that 8-bit PCM is unsigned.

    References
    ----------
    .. [1] IBM Corporation and Microsoft Corporation, "Multimedia Programming
       Interface and Data Specifications 1.0", section "Data Format of the
       Samples", August 1991
       http://www.tactilemedia.com/info/MCI_Control_Info.html

    """
    sampling_rate = signal.sampling_rate
    data = signal.time

    # Reshape to 2D
    data = data.reshape(-1, data.shape[-1])
    warnings.warn("Signal flattened to {data.shape[0]} channels.")

    # Check for .wav file extension
    if filename.split('.')[-1] != 'wav':
        warnings.warn("Extending filename by .wav.")
        filename += '.wav'

    # Check if file exists and for overwrite
    if overwrite is False and os.path.isfile(filename):
        raise FileExistsError(
            "File already exists,"
            "use overwrite option to disable error.")
    else:
        wavfile.write(filename, sampling_rate, data.T)


def read_sofa(filename):
    """
    Import a SOFA file as signal object.

    Parameters
    ----------
    filename : string or open file handle
        Input wav file.

    Returns
    -------
    signal : signal instance
        An audio signal object from the pyfar Signal class
        containing the IR data from the SOFA file with cshape being
        equal to (number of measurements, number of receivers).
    source_coordinates: coordinates instance
        An object from the pyfar Coordinates class containing
        the source coordinates from the SOFA file
        with matching domain, convention and unit.
    receiver_coordinates: coordinates instance
        An object from the pyfar Coordinates class containing
        the receiver coordinates from the SOFA file
        with matching domain, convention and unit.

    Notes
    -----
    * This function is based on the python-sofa package.
    * Only SOFA files of DataType 'FIR' are supported.

    References
    ----------
    .. [1] www.sofaconventions.org
    .. [2] “AES69-2015: AES Standard for File Exchange-Spatial Acoustic Data
       File Format.”, 2015.

    """
    sofafile = sofa.Database.open(filename)
    # Check for DataType
    if sofafile.Data.Type == 'FIR':
        domain = 'time'
        data = np.asarray(sofafile.Data.IR)
        sampling_rate = sofafile.Data.SamplingRate.get_values()
        # Check for units
        if sofafile.Data.SamplingRate.Units != 'hertz':
            raise ValueError(
                "SamplingRate:Units"
                "{sofafile.Data.SamplingRate.Units} is not supported.")
    else:
        raise ValueError("DataType {sofafile.Data.Type} is not supported.")
    signal = Signal(data, sampling_rate, domain=domain)

    # Source
    s_values = sofafile.Source.Position.get_values()
    s_domain, s_convention, s_unit = _sofa_pos(sofafile.Source.Position.Type)
    source_coordinates = Coordinates(
        s_values[:, 0],
        s_values[:, 1],
        s_values[:, 2],
        domain=s_domain,
        convention=s_convention,
        unit=s_unit)
    # Receiver
    r_values = sofafile.Receiver.Position.get_values()
    r_domain, r_convention, r_unit = _sofa_pos(sofafile.Receiver.Position.Type)
    receiver_coordinates = Coordinates(
        r_values[:, 0],
        r_values[:, 1],
        r_values[:, 2],
        domain=r_domain,
        convention=r_convention,
        unit=r_unit)

    return signal, source_coordinates, receiver_coordinates


def _sofa_pos(pos_type):
    if pos_type == 'spherical':
        domain = 'sph'
        convention = 'top_elev'
        unit = 'deg'
    elif pos_type == 'cartesian':
        domain = 'cart'
        convention = 'right'
        unit = 'met'
    else:
        raise ValueError("Position:Type {pos_type} is not supported.")
    return domain, convention, unit


def read(filename):
    """
    Read any compatible pyfar format from disk.

    Parameters
    ----------
    filename : string or open file handle.
        Input file must be haiopy compatible.

    Returns
    -------
    collection: dictionary
        containing PyFar types like
        { 'name1': 'obj1', 'name2': 'obj2' ... }.
    """
    # Check for .far file extension
    if filename.split('.')[-1] != 'far':
        warnings.warn("Extending filename by .far.")
        filename += '.far'

    collection = {}
    with open(filename, 'rb') as f:
        zip_buffer = io.BytesIO()
        zip_buffer.write(f.read())
        with zipfile.ZipFile(zip_buffer) as zip_file:
            zip_paths = zip_file.namelist()
            obj_names = set([path.split('/')[0] for path in zip_paths])
            for name in obj_names:
                json_str = zip_file.read(name + '/json').decode('UTF-8')
                obj_type, obj_dict = json.loads(json_str)
                obj_dict = _decode(obj_dict, zip_file)
                ObjType = _str_to_type(obj_type)
                obj = ObjType._decode(obj_dict)
                if not _is_parfy_type(obj):
                    raise TypeError(
                        f'Objects of type {type(obj)}'
                        'cannot be read from disk.')
                collection[name] = obj

    return collection


def write(filename, compress=False, **objs):
    """
    Write any compatible pyfar format to disk.

    Parameters
    ----------
    filename : string or open file handle.
        Input file must be pyfar compatible.
    compress : bool
        Default is false (uncompressed).
        If false zipfile.ZIP_STORED mode is used,
        if True, zipfile.ZIP_DEFLATED mode is used.
    **objs: named compatible pyfar objects
        - Coordinates
        - Orientations
    """
    # Check for .far file extension
    if filename.split('.')[-1] != 'far':
        warnings.warn("Extending filename by .far.")
        filename += '.far'

    compression = zipfile.ZIP_STORED if compress else zipfile.ZIP_DEFLATED
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", compression) as zip_file:
        for name, obj in objs.items():
            if not _is_parfy_type(obj):
                error = f'Objects of type {type(obj)}'
                'cannot be written to disk.'
                if isinstance(obj, fo.Filter):
                    error = f'{error}. Consider casting to {fo.Filter}'
                raise TypeError(error)

            obj_dict = _encode(
                copy.deepcopy(obj.__dict__), name, zip_file)
            type_obj_pair = [type(obj).__name__, obj_dict]
            zip_file.writestr(f'{name}/json', json.dumps(type_obj_pair))

    with open(filename, 'wb') as f:
        f.write(zip_buffer.getvalue())


def _decode(obj, zipfile):
    """
    This function is exclusively used by `io.read` and enables recursive
    decoding for objects of varying depth.

    Parameters
    ----------
    obj : PyFar-object.

    zipfile: zipfile-object.
        The zipfile object is looped in the recursive structure
        e.g. to decode ndarrays when they occur.
    """
    if isinstance(obj, dict):
        for key in obj.keys():
            _inner_decode(obj, key, zipfile)
    elif isinstance(obj, list):
        for i in range(0, len(obj)):
            _inner_decode(obj, i, zipfile)

    return obj


def _inner_decode(obj, key, zipfile):
    """
    This function is exclusively used by `io._decode`

    Parameters
    ----------
    obj : PyFar-object.

    key :  str or int
        The key provided by the dict or list over which currently is being
        iterated.
    """
    if not _is_type_hint(obj[key]):
        _decode(obj[key], zipfile)
    elif _is_parfy_type(obj[key][0][1:]):
        ParfyType = _str_to_type(obj[key][0][1:])
        obj[key] = ParfyType._decode(obj[key][1])
        _decode(obj[key].__dict__, zipfile)
    elif obj[key][0][1:] == 'dtype':
        obj[key] = getattr(np, obj[key][1])
    elif obj[key][0][1:] == 'ndarray':
        obj[key] = _decode_ndarray(obj[key][1], zipfile)
    else:
        _decode_numpy_scalar(obj, key)
        pass


def _decode_numpy_scalar(obj, key):
    """ This function is exclusively used by `io._inner_decode` and
    decodes numpy scalars e.g. of type `numpy.int32`.
    """
    try:
        numpy_scalar = getattr(np, obj[key][0][1:])
    except AttributeError:
        pass
    else:
        obj[key] = numpy_scalar(obj[key][1])


def _decode_ndarray(obj, zipfile):
    """ This function is exclusively used by `io._inner_decode` and
    decodes `numpy.ndarrays`.
    """
    memfile = io.BytesIO()
    nd_bytes = zipfile.read(obj)
    memfile.write(nd_bytes)
    memfile.seek(0)
    return np.load(memfile, allow_pickle=False)


def _encode(obj, zip_path, zipfile):
    """
    Chooses the right encoding depending on the object type.

    Parameters
    ----------
    obj : PyFar-object.

    zip_path: str.
        zipfile acceps a path-like-string to know where to write
        special types e.g. ndarrays into the archive.

    zipfile: zipfile-object.
        The zipfile object is looped in the recursive structure
        e.g. to encode ndarrays when they occur.

    Returns
    -------
    obj : dict
        A dict derived from the original object that must be  JSON-serializable
        and encodes all not-JSON-serializable objects as:
        (1) A pair of type-hint and value:
            [str, JSON-serializable] e.g. ['$numpy.int32', 42], or
        (2) A pair of ndarray-hint and reference/zip_path:
            [str, str] e.g. ['ndarray', 'my_coordinates/_points']
    """
    if isinstance(obj, dict):
        for key in obj.keys():
            _inner_encode(obj, key, f'{zip_path}/{key}', zipfile)
    elif isinstance(obj, list):
        for i in range(0, len(obj)):
            _inner_encode(obj, i, f'{zip_path}/{i}', zipfile)

    return obj


def _inner_encode(obj, key, zip_path, zipfile):
    if _is_dtype(obj[key]):
        obj[key] = ['$dtype', obj[key].__name__]
    elif isinstance(obj[key], np.ndarray):
        zipfile.writestr(zip_path, _encode_ndarray(obj[key]))
        obj[key] = ['$ndarray', zip_path]
    elif _is_parfy_type(obj[key]):
        obj[key] = [f'${type(obj[key]).__name__}', obj[key].__dict__]
        _encode(obj[key][1], zip_path, zipfile)
    elif _is_numpy_scalar(obj[key]):
        obj[key] = [f'${type(obj[key]).__name__}', obj[key].item()]
    else:
        _encode(obj[key], zip_path, zipfile)


def _encode_ndarray(ndarray):
    """
    The encoding of objects that are composed of primitive and numpy types
    utilizes `obj.__dict__()` and numpy encoding methods.

    Parameters
    ----------
    ndarray: numpy.array.

    Returns
    -------
    bytes.
        They bytes that where written by `numpy.save` into a memfile.

    Notes
    -----
    * Do not allow pickling. It is not safe!
    """
    memfile = io.BytesIO()
    np.save(memfile, ndarray, allow_pickle=False)
    memfile.seek(0)
    return memfile.read()


def _is_parfy_type(obj):
    """ True if object is a Parfy-type.
    """
    type_str = obj if isinstance(obj, str) else type(obj).__name__
    return type_str in [
        'Orientations',
        'Coordinates',
        'Signal',
        'Filter',
        'SphericalVoronoi',
        'NestedDataStruct',
        'MyOtherClass']


def _is_dtype(obj):
    """ True if object is `numpy.dtype`.
    """
    return isinstance(obj, type) and obj.__module__ == 'numpy'


def _is_numpy_scalar(obj):
    """ True if object is any numpy.dtype scalar e.g. `numpy.int32`.
    """
    return type(obj).__module__ == 'numpy'


def _is_type_hint(obj):
    return isinstance(obj, list) \
        and len(obj) == 2 \
        and isinstance(obj[0], str) \
        and obj[0][0] == '$'


def _str_to_type(type_as_string, module='pyfar'):
    """
    Recursively find a PyfarType by passing in a valid type as a string.

    Parameters
    ----------
    type_as_string: string.
        A valid PyfarType.
    module: string.
        Either 'pyfar' or a submodule of pyfar, e.g. 'pyfar.spatial'
        The default is 'pyfar'.

    Returns
    ----------
    PyfarType: type.
        A valid PyfarType.
    """
    try:
        return getattr(sys.modules[module], type_as_string)
    except AttributeError:
        submodules = [
            attrib for attrib in dir(sys.modules[module])
            if not attrib.startswith('__') and attrib.islower()]
    except KeyError:
        return
    for submodule in submodules:
        PyfarType = _str_to_type(
            type_as_string, module=f'{module}.{submodule}')
        if PyfarType:
            return PyfarType

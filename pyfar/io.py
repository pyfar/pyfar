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
from pyfar.spatial.spatial import SphericalVoronoi
from pyfar.utils import str_to_type
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
        containing the IR data from the SOFA file.
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
    loaded_dict: dictionary containing haiopy types.
    """
    obj_dict = {}
    with open(filename, 'rb') as f:
        zip_buffer = io.BytesIO()
        zip_buffer.write(f.read())
        with zipfile.ZipFile(zip_buffer) as zip_file:
            obj_paths = _unpack_zip_paths(zip_file.namelist())
            for obj_name, ndarray_names in obj_paths.items():
                obj_dict[obj_name] = _decode(zip_file, obj_name, ndarray_names)
    return obj_dict


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
    compression = zipfile.ZIP_STORED if compress else zipfile.ZIP_DEFLATED
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", compression) as zip_file:
        for name, obj in objs.items():
            obj_dict_encoded, obj_dict_ndarray = _encode(obj)
            zip_file.writestr(f'{name}/json', json.dumps(obj_dict_encoded))
            for key, value in obj_dict_ndarray.items():
                zip_file.writestr(f'{name}/ndarrays/{key}', value)

    with open(filename, 'wb') as f:
        f.write(zip_buffer.getvalue())


def _decode(zip_file, obj_name, ndarray_names):
    """
    Iterates over object's encoded dictionary and decodes all
    numpy.ndarrays to be prepare object initialization.

    Parameters
    ----------
    obj_dict_encoded: dict.
        Dictionary of encoded compatible haiopy types.

    Returns
    ----------
    obj_dict: dict.
        Decoded dictionary ready for initialization of haiopy types.
    """
    # decoding builtins as json
    json_str = zip_file.read(obj_name + '/json').decode('UTF-8')
    obj_dict = json.loads(json_str)
    # decode numpy.dtypes
    if '_dtype' in obj_dict:
        obj_dict['_dtype'] = getattr(np, obj_dict['_dtype'].split('.')[-1])
    # decoding ndarrays
    for key in ndarray_names:
        memfile = io.BytesIO()
        nd_bytes = zip_file.read(obj_name + '/ndarrays/' + key)
        memfile.write(nd_bytes)
        memfile.seek(0)
        obj_dict[key] = np.load(memfile, allow_pickle=False)
    # build object from obj_dict
    PyfarType = str_to_type(obj_dict['type'])
    if PyfarType == Signal:
        obj = PyfarType(
            obj_dict['_data'],
            obj_dict['_sampling_rate'],
            obj_dict['_n_samples'])
    elif PyfarType == SphericalVoronoi:
        obj = _decode_sphericalvoronoi(obj_dict)
    elif PyfarType in [fo.FilterIIR]:
        obj = _decode_filterFIR(PyfarType, obj_dict)
    else:
        obj = PyfarType()
        del obj_dict['type']
        obj.__dict__.update(obj_dict)
    return obj


def _decode_sphericalvoronoi(obj_dict):    
    sampling = Coordinates(
        obj_dict['sampling'][:, 0],
        obj_dict['sampling'][:, 1],
        obj_dict['sampling'][:, 2],
        domain='cart')
    obj = SphericalVoronoi(
        sampling,
        center=obj_dict['center'])
    return obj


def _decode_filterFIR(PyfarType, obj_dict):
    obj = fo.FilterIIR(
        coefficients=obj_dict['_coefficients'][0, :, :],
        sampling_rate=obj_dict['_sampling_rate'])
    return obj


def _encode(obj):
    """
    Chooses the right encoding depending on the object type.

    Parameters
    ----------
    obj: Compatible Pyfar type.

    Returns
    ----------
    obj_dict_encoded: dict.
        Json compatible dictionary.
    obj_dict_ndarray: dict
        Numpy arrays are not JSON serializable thus encoded differently.
    """
    if type(obj).__name__ == 'NestedDataStruct':
        pass

    pyfarTypes = [
        'Orientations',
        'Coordinates',
        'Signal',
        'SphericalVoronoi',
        'Filter',
        'FilterIIR',
        'FilterFIR',
        'NestedDataStruct']
    if not any(pyfarType == type(obj).__name__ for pyfarType in pyfarTypes):
        raise TypeError(
            f'Objects of type {type(obj)} cannot be written to disk')

    if isinstance(obj, SphericalVoronoi):
        return _encode_sphericalvoronoi(obj)

    # if isinstance(obj, fo.Filter) or issubclass(obj, fo.Filter):
    #     return _encode_filter(obj)

    return _encode_obj_by_dict(obj)


def pyfar_types():
    return [
        'Orientations',
        'Coordinates',
        'Signal',
        'SphericalVoronoi',
        'Filter',
        'FilterIIR',
        'FilterFIR',
        'NestedDataStruct']


def encode_recursively(node, name):
    if any(pyfarType == type(node).__name__ for pyfarType in pyfar_types()):
        node = node.__dict__

    if isinstance(node, dict):
        for key, value in node.items():
            pass
    elif isinstance(node, list):
        pass
    elif not type(node).__name__ != 'builtins':
        pass
    # The actual encoding

    pass


def _encode_obj_by_dict(obj):
    """
    The encoding of objects that are composed of primitive and numpy types
    utilizes `obj.__dict__()` and numpy encoding methods.

    Parameters
    ----------
    obj: Compatible Pyfar type.

    Returns
    ----------
    obj_dict_encoded: dict.
        Json compatible dictionary.
    obj_dict_ndarray: dict
        Numpy arrays are not JSON serializable thus encoded differently.
    """
    obj_dict_encoded = copy.deepcopy(obj.__dict__)
    obj_dict_ndarray = {}
    for key, value in obj.__dict__.items():
        if isinstance(value, np.ndarray):
            obj_dict_ndarray[key] = _encode_ndarray(value)
            del obj_dict_encoded[key]
        elif isinstance(value, type) and value.__module__ == 'numpy':
            obj_dict_encoded[key] = np.dtype(value).name
        elif (type(value) is not type and type(value).__module__ != 'builtins'
            or isinstance(value, dict)):
            del obj_dict_encoded[key]

    obj_dict_encoded['type'] = type(obj).__name__

    return obj_dict_encoded, obj_dict_ndarray


def _encode_sphericalvoronoi(obj):
    """
    The encoding of objects that are composed of primitive and numpy types
    utilizes `obj.__dict__()` and numpy encoding methods.

    Parameters
    ----------
    obj: Compatible Pyfar type.

    Returns
    ----------
    obj_dict_encoded: dict.
        Json compatible dictionary.
    obj_dict_ndarray: dict
        Numpy arrays are not JSON serializable thus encoded differently.
    """
    obj_dict_encoded = {}
    obj_dict_ndarray = {}
    obj_dict_encoded['type'] = type(obj).__name__
    obj_dict_ndarray['sampling'] = _encode_ndarray(obj.points)
    obj_dict_ndarray['center'] = _encode_ndarray(obj.center)
    return obj_dict_encoded, obj_dict_ndarray


def _encode_filter(obj):
    warnings.warn(f'`io.write` writing object of type {type(obj)}: ' 
        'It is not possible to save `filter_func` to disk.')
    obj_dict_encoded = {}
    obj_dict_ndarray = {}
    obj_dict_encoded['type'] = type(obj).__name__
    obj_dict_ndarray['_coefficients'] = _encode_ndarray(
        obj.__dict__['_coefficients'])
    obj_dict_encoded['_sampling_rate'] = obj.__dict__['_sampling_rate']
    obj_dict_encoded['_comment'] = obj.__dict__['_comment']
    obj_dict_ndarray['_state'] = _encode_ndarray(obj.__dict__['_state'])
    return obj_dict_encoded, obj_dict_ndarray
    

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


def _unpack_zip_paths(zip_paths):
    """
    This is a helper function for read() to unpack zip-paths,
    e.g. 'coords/ndarrays/_points' and orientations/ndarrays/_quat
    becomes {'coords': [_points], 'orientations': '_quat'}

    Parameters
    ----------
    zip_paths: list of strings.
        Retreived zipfile.ZipFile.namelist()

    Returns
    ----------
    obj_paths: dict.
        Dictionary that contains unpacked zip paths.
    """
    obj_paths = {}
    for path in zip_paths:
        paths = path.split('/')
        obj_paths.setdefault(paths[0], [])
        if paths[1] == 'ndarrays':
            obj_paths[paths[0]].append(paths[2])
    return obj_paths

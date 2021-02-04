import scipy.io.wavfile as wavfile
import os.path
import warnings
import numpy as np
import sofa
import json
import copy
import io
import sys

from pyfar import Signal
from pyfar import Coordinates
from pyfar.spatial.spatial import SphericalVoronoi
from pyfar.utils import str_to_type
import pyfar.dsp.classes as fo


def _decode(obj, zipfile):
    if isinstance(obj, dict):
        for key in obj.keys():
            _inner_decode(obj, key, zipfile)
    elif isinstance(obj, list):
        for i in range(0, len(obj)):
            _inner_decode(obj, i, zipfile)

    return obj


def _inner_decode(obj, key, zipfile):
    if not _is_type_hint(obj[key]):
        _decode(obj[key], zipfile)
    elif _is_mylib_type(obj[key][0]):
        MyType = str_to_type(obj[key][0])
        obj[key] = MyType._decode(obj[key][1])
    elif obj[key][0] == 'dtype':
        obj[key] = getattr(np, obj[key][1])
    elif obj[key][0] == 'ndarray':
        obj[key] = _decode_ndarray(obj[key][1], zipfile)


def _decode_ndarray(obj, zipfile):
    memfile = io.BytesIO()
    nd_bytes = zipfile.read(obj)
    memfile.write(nd_bytes)
    memfile.seek(0)
    return np.load(memfile, allow_pickle=False)


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


def _encode(obj, zip_path, zipfile):
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
    if isinstance(obj, dict):
        for key in obj.keys():
            _inner_encode(obj, key, f'{zip_path}/{key}', zipfile)
    elif isinstance(obj, list):
        for i in range(0, len(obj)):
            _inner_encode(obj, i, f'{zip_path}/{i}', zipfile)

    return obj


def _inner_encode(obj, key, zip_path, zipfile):
    if _is_dtype(obj[key]):
        obj[key] = ['dtype', obj[key].__name__]
    elif isinstance(obj[key], np.ndarray):
        zipfile.writestr(zip_path, _encode_ndarray(obj[key]))
        obj[key] = ['ndarray', zip_path]
    elif _is_mylib_type(obj[key]):
        obj[key] = [type(obj[key]).__name__, obj[key].__dict__]
        _encode(obj[key][1], zip_path, zipfile)
    else:
        _encode(obj[key], zip_path, zipfile)


def _is_mylib_type(obj):
    type_str = obj if isinstance(obj, str) else type(obj).__name__
    return type_str in [
        'NestedDataStruct', 'MyOtherClass']


def _is_dtype(obj):
    return isinstance(obj, type) and obj.__module__ == 'numpy'


def _is_type_hint(obj):
    return isinstance(obj, list) and len(obj) == 2


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


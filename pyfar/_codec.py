"""
Brief
=====

This module is not part of the public API. It contains encoding and decoding
functionality which is exclusively used by `io.write` and `io.read`. It enables
storing and transmitting Pyfar-Objects without using the unsafe pickle
protocoll.

Design and Function
===================
The `_encode` and `_decode` functions are entry points for an algorithm
that recursively processes Parfy-objects of varying depths. The result can be
stored in a zipfile or similar structure.

There are three basic encoding/decoding types:

    (1) Builtins types are directly written in a JSON-format.

    (2) Types that can be trivially derived from builtins are first
        cast into their builtin-counterparts. The result and a type-hint
        are put into a pair of list-type and written into the same JSON-string.
        The pair looks as follows:
            [str, builtin] e.g. ['$int64', 42]

    (2) Types that cannot be easily derived from builtins, such as
        numpy.ndarrays, are encoded separately with a dedicated function like
        `_encode_ndarray`. The result is written to a dedicated path in the
        zip-archive. This zip-path is stored as a reference together with a
        type-hint as a pair into the JSON-form
            [str, str] e.g. ['$ndarray', '/my_obj/_signal']

Class-Level
===========

Actually pyfar_obj.__dict__ could be passed to `_encode` or `_decode`
respectively. Some objects however need special treatment at the highest level
of encoding and decoding. Therefore `PyfarClass._encode` and
`PyfarClass._decode` must be implemented in each class. E.g.:

class MyClass:

    def _encode(self):
        return self.copy().__dict__

    @classmethod
    def _decode(cls, obj_dict):
        obj = cls()
        obj.__dict__.update(obj_dict)
        return obj

Data Inspection
===============

Once data is written to disk you can rename the file-extension to .zip, open
and inspect the archive. The JSON-file in the archive must reflect the
data structure, e.g. like this:

JSON
----
{
    "_n": 42,
    "_comment": "My String",
    "_subobj": {
        "signal": [
            "$ndarray",
            "my_obj/_subobj/signal"
        ],
        "_m": 49
    }
    "_list": [
        1,
        [
            "$dtype",
            "int32"
        ],
    ]
}
----

"""

import io
import sys
import numpy as np


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
    elif _is_pyfar_type(obj[key][0][1:]):
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
    elif _is_pyfar_type(obj[key]):
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


def _is_pyfar_type(obj):
    """ True if object is a Parfy-type.
    """
    type_str = obj if isinstance(obj, str) else type(obj).__name__
    return type_str in [
        'Orientations',
        'Coordinates',
        'Signal',
        'Filter',
        'SphericalVoronoi']


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

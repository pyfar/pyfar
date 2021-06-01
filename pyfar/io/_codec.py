"""
Brief
=====

This module is not part of the public API. It contains encoding and decoding
functionality which is exclusively used by `io.write` and `io.read`. It enables
storing and transmitting Pyfar- and Numpy-objects without using the unsafe
pickle protocoll.

Design and Function
===================

The `_encode` and `_decode` functions are entry points for an algorithm
that recursively processes data structures of varying depths. The result is
stored in a zipfile or similar structure.

Data structures are decomposed into one of the following
three basic encoding/decoding types:

    (1) Builtins types are directly written in a JSON-format.

    (2) Types that can be trivially derived from builtins are first
        cast into their builtin-counterparts. The result and a type-hint
        are put into a list-type pair of type hint and object and written into
        the same JSON-string. The pair looks as follows:
            [str, builtin] e.g. ['$int64', 42]

    (3) Types that cannot be easily derived from builtins, such as
        numpy.ndarrays, are encoded separately with a dedicated function like
        `_encode_ndarray`. The result is written to a dedicated path in the
        zip-archive. This zip-path is stored as a reference together with a
        type-hint as a pair into the JSON-form
            [str, str] e.g. ['$ndarray', '/my_obj/_signal']

Numpy-types can be stored directly in the zipfile. In this case type hints,
such as `$ndarray`, become the name of the file in the archive.


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
data structure of an object, e.g. like this:

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

Names, Type Hints and Zippath
=============================

    (1) Object names are the keys of **objs in `io.write` and only
        exist at the first hierarchical layer of zip_paths ([0])

    (2) Object hints are e.g. `$Coordinates` for Pyfar-objects, `$ndarray` etc.
        and exist only at the second layer of zip_paths ([1])

    E.g. in a single zip_path: `MyMicrophonePositions/$Coordinates`
    - `MyMicrophonePositions` is the name of the object and
    - `$Coordinates` is the type hint.

"""

import io
import sys
import json
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
    elif any([isinstance(obj, x) for x in [list, tuple, set, frozenset]]):
        for i in range(0, len(obj)):
            _inner_decode(obj, i, zipfile)

    return obj


def _inner_decode(obj, key, zipfile):
    """
    This function is exclusively used by `_codec._encode` and casts the obj
    in case it was not JSON-serializable back into ther original type
    e.g. by using a typehint (str) like: '$ndarray', '$tuple' etc.

    If the obj is nested, _innner_decode goes one level deeper by reentering
    _decode again.

    Parameters
    ----------
    obj : PyFar-object.

    key :  str or int
        The key provided by the dict or list over which currently is being
        iterated.

    zipfile: zipfile
    """
    if not _is_type_hint(obj[key]):
        _decode(obj[key], zipfile)
    elif _is_pyfar_type(obj[key][0][1:]):
        PyfarType = _str_to_type(obj[key][0][1:])
        obj[key] = PyfarType._decode(obj[key][1])
        _decode(obj[key].__dict__, zipfile)
    elif obj[key][0][1:] == 'dtype':
        obj[key] = getattr(np, obj[key][1])
    elif obj[key][0][1:] == 'ndarray':
        obj[key] = _decode_ndarray(obj[key][1], zipfile)
    elif obj[key][0][1:] == 'complex':
        obj[key] = complex(obj[key][1][0], obj[key][1][1])
    elif obj[key][0][1:] == 'tuple':
        obj[key] = tuple(obj[key][1])
    elif obj[key][0][1:] == 'set':
        obj[key] = set(tuple(obj[key][1]))
    elif obj[key][0][1:] == 'frozenset':
        obj[key] = frozenset(tuple(obj[key][1]))
    else:
        _decode_numpy_scalar(obj, key)


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
    decodes `numpy.ndarrays` from a memfile.
    """
    # Numpy.load is applied on a memory file instead of a physical file
    memfile = io.BytesIO()
    nd_bytes = zipfile.read(obj)
    memfile.write(nd_bytes)
    memfile.seek(0)
    return np.load(memfile, allow_pickle=False)


def _decode_object_json_aided(name, type_hint, zipfile):
    """
    Decodes composed objects with the help of JSON.

    Parameters
    ----------
    name: str
        The object name, usually keys from **objs, see `io.write`.
    type_hint: str
        The object's type hint, starts with '$'.
    zipfile: zipfile
        The zipfile from where we'd like to read data.
    """
    json_str = zipfile.read(f'{name}/{type_hint}').decode('UTF-8')
    obj_dict_encoded = json.loads(json_str)
    obj_dict = _decode(obj_dict_encoded, zipfile)
    ObjType = _str_to_type(type_hint[1:])
    try:
        return ObjType._decode(obj_dict)
    except AttributeError:
        raise NotImplementedError(
            f'You must implement `{type}._decode` first.')


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
            [str, JSON-serializable] e.g.
            ['$numpy.int32', 42], ['$tuple', [1, 2, 3]]
            or
        (2) A pair of ndarray-hint and reference/zip_path:
            [str, str] e.g. ['ndarray', 'my_coordinates/_points']
    """
    if isinstance(obj, dict):
        for key in obj.keys():
            _inner_encode(obj, key, f'{zip_path}/{key}', zipfile)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for i in range(0, len(obj)):
            _inner_encode(obj, i, f'{zip_path}/{i}', zipfile)

    return obj


def _inner_encode(obj, key, zip_path, zipfile):
    """
    This function is exclusively used by `_codec._encode` and casts the obj
    in case it is not JSON-serializable into a proper format for the zipfile
    e.g. by adding a typehint (str) like: '$ndarray', '$tuple' etc. and
    a proper zip path/reference if necessary like it's the case for ndarrays.

    If the obj is nested, _innner_ecode goes one level deeper by reentering
    _encode again.

    Parameters
    ----------
    obj : PyFar-object.

    key :  str or int
        The key provided by the dict or list over which currently is being
        iterated.

    zip_path: str
        The potential zip path looped through all recursions.

    zipfile: zipfile
    """
    if _is_dtype(obj[key]):
        obj[key] = ['$dtype', obj[key].__name__]
    elif isinstance(obj[key], np.ndarray):
        zipfile.writestr(zip_path, _encode_ndarray(obj[key]))
        obj[key] = ['$ndarray', zip_path]
    elif _is_pyfar_type(obj[key]):
        obj[key] = [f'${type(obj[key]).__name__}', obj[key]._encode()]
        _encode(obj[key][1], zip_path, zipfile)
    elif _is_numpy_scalar(obj[key]):
        obj[key] = [f'${type(obj[key]).__name__}', obj[key].item()]
    elif isinstance(obj[key], complex):
        obj[key] = ['$complex', [obj[key].real, obj[key].imag]]
    elif isinstance(obj[key], (tuple, set, frozenset)):
        obj[key] = [f'${type(obj[key]).__name__ }', list(obj[key])]
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

    Note
    ----
    * Do not allow pickling. It is not safe!
    """
    # `Numpy.save` is applied on a memory file instead of a physical file
    memfile = io.BytesIO()
    np.save(memfile, ndarray, allow_pickle=False)
    memfile.seek(0)
    return memfile.read()


def _encode_object_json_aided(obj, name, zipfile):
    """
    Encodes composed objects with the help of JSON.

    Parameters
    ----------
    obj: PyFar-type
        The object, usually values from **objs, see `io.write`.
    name: str
        The object's name, usually keys from **objs, see `io.write`.
    zipfile: zipfile
        The zipfile where we'd like to write data.
    """
    try:
        obj_dict = _encode(obj._encode(), name, zipfile)
        type_hint = f'${type(obj).__name__}'
        zipfile.writestr(
            f'{name}/{type_hint}',
            json.dumps(obj_dict))
    except AttributeError:
        raise NotImplementedError(
            f'You must implement `{type}._encode` first.')


def _is_pyfar_type(obj):
    """ True if object is a Pyfar-type.
    """
    type_str = obj if isinstance(obj, str) else type(obj).__name__
    return type_str in [
        'Orientations',
        'Coordinates',
        'Signal',
        'Filter',
        'FilterFIR',
        'FilterIIR',
        'FilterSOS',
        'SphericalVoronoi',
        'TimeData',
        'FrequencyData']


def _is_numpy_type(obj):
    """ True if object is a Numpy-type.
    """
    return type(obj).__module__ == np.__name__


def _is_dtype(obj):
    """ True if object is `numpy.dtype`.
    """
    return isinstance(obj, type) and (
        obj.__module__ == 'numpy' or obj == complex)


def _is_numpy_scalar(obj):
    """ True if object is any numpy.dtype scalar e.g. `numpy.int32`.
    """
    return type(obj).__module__ == 'numpy'


def _is_type_hint(obj):
    """ Check if object is stored along with its type in the typical format:
            [str, str] => [typehint, value] e.g. ['$complex', (3 + 4j)]
    """
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

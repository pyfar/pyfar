"""
Read and write objects to disk, read and write WAV files, read SOFA files.

The functions :py:func:`read` and :py:func:`write` allow to save or load
several pyfar objects and other variables. So, e.g., workspaces in notebooks
can be stored. :py:class:`Signal <pyfar.signal.Signal>` objects can be
imported and exported as WAV files using :py:func:`read_wav` and
:py:func:`write_wav`. :py:func:`read_sofa` provides functionality to read the
data stored in a SOFA file.
"""
import os.path
import pathlib
import warnings
import numpy as np
import sofa
import zipfile
import io
import soundfile as sf


from pyfar import Signal
from pyfar import Coordinates

from . import _codec as codec
import pyfar.classes.filter as fo


def read_sofa(filename):
    """
    Import a SOFA file as :py:class:`~pyfar.classes.audio.Signal` object.

    Parameters
    ----------
    filename : string, Path
        Input SOFA file (cf. [#]_, [#]_).

    Returns
    -------
    signal : Signal
        :py:class:`~pyfar.classes.audio.Signal` object containing the data
        stored in `SOFA_Object.Data.IR`.
        `cshape` is equal to ``(number of measurements, number of receivers)``.
    source_coordinates : Coordinates
        Coordinates object containing the data stored in
        `SOFA_object.SourcePosition`. The domain, convention and unit are
        automatically matched.
    receiver_coordinates : Coordinates
        Coordinates object containing the data stored in
        `SOFA_object.RecevierPosition`. The domain, convention and unit are
        automatically matched.

    Notes
    -----
    * This function is based on the python-sofa [#]_.
    * Currently, only SOFA files of `DataType` ``FIR`` are supported.

    References
    ----------
    .. [#] https://www.sofaconventions.org
    .. [#] “AES69-2015: AES Standard for File Exchange-Spatial Acoustic Data
        File Format.”, 2015.
    .. [#] https://github.com/spatialaudio/python-sofa

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
    Read any compatible pyfar object or numpy array (.far file) from disk.

    Parameters
    ----------
    filename : string, Path
        Input file. If no extension is provided, .far-suffix is added.

    Returns
    -------
    collection: dict
        Contains pyfar objects like
        ``{ 'name1': 'obj1', 'name2': 'obj2' ... }``.

    Examples
    --------
    Read signal and orientations objects stored in a .far file.

    >>> collection = pyfar.read('my_objs.far')
    >>> my_signal = collection['my_signal']
    >>> my_orientations = collection['my_orientations']
    """
    # Check for .far file extension
    filename = pathlib.Path(filename).with_suffix('.far')

    collection = {}
    with open(filename, 'rb') as f:
        zip_buffer = io.BytesIO()
        zip_buffer.write(f.read())
        with zipfile.ZipFile(zip_buffer) as zip_file:
            zip_paths = zip_file.namelist()
            obj_names_hints = [
                path.split('/')[:2] for path in zip_paths if '/$' in path]
            for name, hint in obj_names_hints:
                if codec._is_pyfar_type(hint[1:]):
                    obj = codec._decode_object_json_aided(name, hint, zip_file)
                elif hint == '$ndarray':
                    obj = codec._decode_ndarray(f'{name}/{hint}', zip_file)
                else:
                    raise TypeError(
                        '.far-file contains unknown types.'
                        'This might occur when writing and reading files with'
                        'different versions of Pyfar.')
                collection[name] = obj

        if 'builtin_wrapper' in collection:
            for key, value in collection['builtin_wrapper'].items():
                collection[key] = value
            collection.pop('builtin_wrapper')

    return collection


def write(filename, compress=False, **objs):
    """
    Write any compatible pyfar object or numpy array and often used builtin
    types as .far file to disk.

    Parameters
    ----------
    filename : string
        Full path or filename. If now extension is provided, .far-suffix
        will be add to filename.
    compress : bool
        Default is ``False`` (uncompressed).
        Compressed files take less disk space but need more time for writing
        and reading.
    **objs:
        Objects to be saved as key-value arguments, e.g.,
        ``name1=object1, name2=object2``.

    Examples
    --------

    Save Signal object, Orientations objects and numpy array to disk.

    >>> s = pyfar.Signal([1, 2, 3], 44100)
    >>> o = pyfar.Orientations.from_view_up([1, 0, 0], [0, 1, 0])
    >>> a = np.array([1,2,3])
    >>> pyfar.io.write('my_objs.far', signal=s, orientations=o, array=a)

    Notes
    -----
    * Supported builtin types are:
      bool, bytes, complex, float, frozenset, int, list, set, str and tuple
    """
    # Check for .far file extension
    filename = pathlib.Path(filename).with_suffix('.far')
    compression = zipfile.ZIP_STORED if compress else zipfile.ZIP_DEFLATED
    zip_buffer = io.BytesIO()
    builtin_wrapper = codec.BuiltinsWrapper()
    with zipfile.ZipFile(zip_buffer, "a", compression) as zip_file:
        for name, obj in objs.items():
            if codec._is_pyfar_type(obj):
                codec._encode_object_json_aided(obj, name, zip_file)
            elif codec._is_numpy_type(obj):
                codec._encode({f'${type(obj).__name__}': obj}, name, zip_file)
            elif type(obj) in codec._supported_builtin_types():
                builtin_wrapper[name] = obj
            else:
                error = (
                    f'Objects of type {type(obj)} cannot be written to disk.')
                if isinstance(obj, fo.Filter):
                    error = f'{error}. Consider casting to {fo.Filter}'
                raise TypeError(error)

        if len(builtin_wrapper) > 0:
            codec._encode_object_json_aided(
                builtin_wrapper, 'builtin_wrapper', zip_file)

    with open(filename, 'wb') as f:
        f.write(zip_buffer.getvalue())


def read_audio(filename, dtype='float64', **kwargs):
    """
    Import an audio file as :py:class:`~pyfar.classes.audio.Signal` object.

    Parameters
    ----------
    filename : string, Path
        Input file.
    dtype : {'float64', 'float32', 'int32', 'int16'}, optional
        Data type of the returned signal, by default ``'float64'``.
        Floating point audio data is typically in the range from
        ``-1.0`` to ``1.0``.  Integer data is in the range from
        ``-2**15`` to ``2**15-1`` for ``'int16'`` and from ``-2**31`` to
        ``2**31-1`` for ``'int32'``.
    **kwargs
        Other keyword arguments to be passed to ``soundfile.read``. This is
        needed, e.g, to read RAW audio files.


    Returns
    -------
    signal : Signal
        :py:class:`~pyfar.classes.audio.Signal` object containing the audio
        data.

    Notes
    -----
    * This function is based on ``soundfile.read``.
    * Reading int values from a float file will *not* scale the data to
      [-1.0, 1.0). If the file contains ``np.array([42.6], dtype='float32')``,
      you will read ``np.array([43], dtype='int32')`` for ``dtype='int32'``.
    """
    data, sampling_rate = sf.read(
        file=filename, dtype=dtype, always_2d=True, **kwargs)
    signal = Signal(data.T, sampling_rate, domain='time', dtype=dtype)
    return signal


def write_audio(signal, filename, subtype=None, overwrite=True, **kwargs):
    """
    Write a :py:class:`~pyfar.classes.audio.Signal` object as a audio file to
    disk.

    Parameters
    ----------
    signal : Signal
        Object to be written.
    filename : string, Path
        Output file. The format is determined from the file extension.
        See :py:func:`available_formats` for all possible formats.
    subtype : str, optional
        The subtype of the sound file, the default value depends on the
        selected `format` (see :py:func:`default_subtype`).
        See :py:func:`available_subtypes` for all possible subtypes for
        a given ``format``.
    overwrite : bool
        Select wether to overwrite the audio file, if it already exists.
        The default is ``True``.
    **kwargs
        Other keyword arguments to be passed to ``soundfile.write``.

    Notes
    -----
    * Signals are flattened before writing to disk (e.g. a signal with
      ``cshape = (3, 2)`` will be written to disk as a six channel audio file).
    * This function is based on ``soundfile.write``.
    * Amplitudes larger than +/- 1 are clipped.

    """
    sampling_rate = signal.sampling_rate
    data = signal.time

    # Reshape to 2D
    data = data.reshape(-1, data.shape[-1])
    if len(signal.cshape) != 1:
        warnings.warn(f"Signal flattened to {data.shape[0]} channels.")

    # Check if file exists and for overwrite
    if overwrite is False and os.path.isfile(filename):
        raise FileExistsError(
            "File already exists,"
            "use overwrite option to disable error.")
    else:
        sf.write(
            file=filename, data=data.T, samplerate=sampling_rate,
            subtype=subtype, **kwargs)


def read_wav(filename):
    """
    Import a WAV file as :py:class:`~pyfar.classes.audio.Signal` object.

    Parameters
    ----------
    filename : string, Path
        Input file.

    Returns
    -------
    signal : Signal
        :py:class:`~pyfar.classes.audio.Signal` object containing the audio
        data from the WAV file.

    Notes
    -----
    * This function is based on :py:func:`read_audio`.
    """
    signal = read_audio(filename)
    return signal


def write_wav(signal, filename, subtype=None, overwrite=True):
    """
    Write a :py:class:`~pyfar.classes.audio.Signal` object as a WAV file to
    disk.

    Parameters
    ----------
    signal : Signal
        Object to be written.
    filename : string, Path
        Output file.
    overwrite : bool
        Select wether to overwrite the WAV file, if it already exists.
        The default is ``True``.

    Notes
    -----
    * Signals are flattened before writing to disk (e.g. a signal with
      ``cshape = (3, 2)`` will be written to disk as a six channel wav file).
    * This function is based on :py:func:`write_audio`.
    * Amplitudes larger than +/- 1 are clipped.

    """
    # .wav file extension
    filename = pathlib.Path(filename).with_suffix('.wav')

    write_audio(signal, filename, subtype=subtype, overwrite=overwrite)


def available_formats():
    """Return a dictionary of available audio formats.

    Notes
    -----
    This function is a wrapper of ``soundfile.available_formats()``.

    Examples
    --------
    >>> import pyfar as pf
    >>> pf.available_formats()
    {'FLAC': 'FLAC (FLAC Lossless Audio Codec)',
     'OGG': 'OGG (OGG Container format)',
     'WAV': 'WAV (Microsoft)',
     'AIFF': 'AIFF (Apple/SGI)',
     ...
     'WAVEX': 'WAVEX (Microsoft)',
     'RAW': 'RAW (header-less)',
     'MAT5': 'MAT5 (GNU Octave 2.1 / Matlab 5.0)'}

    """
    return sf.available_formats()


def available_subtypes(format=None):
    """Return a dictionary of available audio subtypes.

    Parameters
    ----------
    format : str
        If given, only compatible subtypes are returned.

    Notes
    -----
    This function is a wrapper of ``soundfile.available_subtypes()``.

    Examples
    --------
    >>> import pyfar as pf
    >>> pf.available_subtypes('FLAC')
    {'PCM_24': 'Signed 24 bit PCM',
     'PCM_16': 'Signed 16 bit PCM',
     'PCM_S8': 'Signed 8 bit PCM'}

    """
    return sf.available_subtypes(format=format)


def default_subtype(format):
    """Return the default subtype for a given format.

    Notes
    -----
    This function is a wrapper of ``soundfile.default_subtype()``.

    Examples
    --------
    >>> import soundfile as sf
    >>> sf.default_subtype('WAV')
    'PCM_16'
    >>> sf.default_subtype('MAT5')
    'DOUBLE'

    """
    return sf.default_subtype(format)

"""
This module contains anechoic audio content and impulse responses for listening
and illustration. Note that each file has a separate license that is listed
below.

Quick listening is, e.g., possible with `sounddevice
<https://python-sounddevice.readthedocs.io>`_ installed:

>>> import pyfar as pf
>>> import sounddevice as sd
>>> # Load, illustrate and play speech signal
>>> speech = pf.signals.files.speech()
>>> pf.plot.spectrogram(speech)
>>> sd.play(speech.time.T, speech.sampling_rate)
"""
import os
import numpy as np
import urllib3
import pyfar as pf

# path for saving/reading files
file_dir = os.path.join(os.path.dirname(__file__), 'files')


def castanets(sampling_rate=44100):
    """
    Get an anechoic castanet sample.

    Castanets rhythm from EUB SQAM CD track 27 re-programmed as anechoic
    version using samples from the Vienna Symphonic Library [#]_.

    .. note ::

        **License**: CC 0, Matthias Frank

    Parameters
    ----------
    sampling_rate : int, optional
        The sampling rate of the sample in Hz. The default of ``44100`` uses
        the sample as it is, any other value uses
        :py:func:`~pyfar.dsp.resample` for resampling to the desired
        sampling rate.

    Returns
    -------
    castanets : Signal
        The castanets sample.

    References
    ----------
    .. [#] https://iaem.at/Members/frank/sounds/
    """

    # download files if requires
    files = _load_files('castanets')

    # load castanets
    castanets = pf.io.read_audio(os.path.join(file_dir, files[0]))

    castanets.fft_norm = "rms"

    # resample castanets
    if sampling_rate != 44100:
        castanets = pf.dsp.resample(castanets, sampling_rate, post_filter=True)

    return castanets


def drums(sampling_rate=48000):
    """
    Get a dry drum sample.

    The sample was recorded with microphones close to the drums in a dry
    rehearsal room.

    .. note ::

        **License**: CC BY Fabian Brinkmann, Johannes M. Arend

    Parameters
    ----------
    sampling_rate : int, optional
        The sampling rate of the sample in Hz. The default of ``44100`` uses
        the sample as it is, any other value uses
        :py:func:`~pyfar.dsp.resample` for resampling to the desired
        sampling rate.

    Returns
    -------
    drums : Signal
        The drum sample.
    """

    # download files if requires
    files = _load_files('drums')

    # load drums
    drums = pf.io.read_audio(os.path.join(file_dir, files[0]))
    drums.fft_norm = "rms"

    # level to make sure all contents have approximately the same loudness
    drums.time *= .9

    # resample drums
    if sampling_rate != 48000:
        drums = pf.dsp.resample(drums, sampling_rate, post_filter=True)

    return drums


def guitar(sampling_rate=48000):
    """
    Get an anechoic guitar sample.

    The data is an excerpt from the file `Flamenco2_U89.wav` from the Cologne
    University of Applied Sciences, Anechoic Recordings  [#]_.

    .. note ::

        **License**: CC BY-SA Michio Woirgard, Philipp Stade, Jeffrey Amankwor,
        Benjamin Bernschütz and Johannes Arend, Audio Group, Cologne University
        of Applied Sciences

    Parameters
    ----------
    sampling_rate : int, optional
        The sampling rate of the sample in Hz. The default of ``48000`` uses
        the sample as it is, any other value uses
        :py:func:`~pyfar.dsp.resample` for resampling to the desired
        sampling rate.

    Returns
    -------
    guitar : Signal
        The guitar sample.

    References
    ----------
    .. [#] http://audiogroup.web.th-koeln.de/anechoic.html
    """

    # download files if requires
    files = _load_files('guitar')

    # load guitar
    guitar = pf.io.read_audio(os.path.join(file_dir, files[0]))
    guitar.fft_norm = "rms"

    # level to make sure all contents have approximately the same loudness
    guitar.time *= .6

    # resample guitar
    if sampling_rate != 48000:
        guitar = pf.dsp.resample(guitar, sampling_rate, post_filter=True)

    return guitar


def speech(voice="female", sampling_rate=44100):
    """
    Get an anechoic speech sample.

    The samples were taken from 'Music for Archimedes' [#]_ (Tracks 4, 5) with
    kind permission of Bang & Olufsen for research and personal purposes. Any
    commercial use or publication requires approval of Bang & Olufsen. For more
    information on the recordings see [#]_.

    .. note ::

        **Copyright**: Bang & Olufsen

    Parameters
    ----------
    voice : str, optional
        Choose between a ``'female'`` (default) and ``'male'`` voice.
    sampling_rate : int, optional
        The sampling rate of the sample in Hz. The default of ``44100`` uses
        the sample as it is, any other value uses
        :py:func:`~pyfar.dsp.resample` for resampling to the desired
        sampling rate.

    Returns
    -------
    speech : Signal
        The speech sample.

    References
    ----------
    .. [#] Music for Archimedes. Bang & Olufsen, 1992, CD B&O 101.

    .. [#] V. Hansen, and G. Munch, 'Making Recordings for Simulation Tests in
           the Archimedes Project,' J. Audio Eng. Soc. 39, 768–774 (1991).
    """

    # download files if requires
    files = _load_files('speech')

    # load speech
    file = files[0] if voice == "female" else files[1]
    speech = pf.io.read_audio(os.path.join(file_dir, file))
    speech.fft_norm = "rms"

    # level to make sure all contents have approximately the same loudness
    gain = 0.5 if voice == "female" else 0.3
    speech.time *= gain

    # resample speech
    if sampling_rate != 44100:
        speech = pf.dsp.resample(speech, sampling_rate, post_filter=True)

    return speech


def binaural_room_impulse_response(
        diffuse_field_compensation=False, sampling_rate=48000):
    """
    Get a binaural room impulse response (BRIR).

    The BRIR was recorded with the FABIAN head and torso simulator in the
    Berliner Philharmonie [#]_ (Emitter 17). The head of FABIAN was rotated
    25 degree to the right. For more information see [#]_. A matching room
    impulse response can be obtained by :py:func:`~room_impulse_response`.

    .. note ::

        **License**: CC BY-NC-SA 4.0, David Ackermann, Audio Communication
        Group, Technical University of Berlin

    Parameters
    ----------
    diffuse_field_compensation : bool, optional
        Apply a diffuse field compensation to the BRIR. This can be used as a
        simple headphone compensation filter when listening to the BRIR. The
        default is ``False``, which does not apply the compensation. The
        diffuse field compensation is taken from
        :py:func:`~head_related_impulse_responses`
    sampling_rate : int, optional
        The sampling rate of the BRIR in Hz. The default of ``48000`` uses the
        BRIR as it is, any other value uses :py:func:`~pyfar.dsp.resample`
        for resampling to the desired sampling rate.

    Returns
    -------
    brir : Signal
        The BRIR

    References
    ----------
    .. [#] http://dx.doi.org/10.14279/depositonce-15774

    .. [#] D. Ackermann, J. Domann, F. Brinkmann, J. M. Arend, M. Schneider,
           C. Pörschmann, and S. Weinzierl 'Recordings of a Loudspeaker
           Orchestra with Multi-Channel Microphone Arrays for the Evaluation of
           Spatial Audio Methods,' J. Audio Eng. Soc. (submitted)
    """

    # download files if requires
    files = _load_files('binaural_room_impulse_response')
    if diffuse_field_compensation:
        files_2 = _load_files('head_related_impulse_responses')

    # load brir
    brir = pf.io.read_audio(os.path.join(file_dir, files[0]))

    # load and resample diffuse field filter
    if diffuse_field_compensation:
        inverse_ctf, *_ = pf.io.read_sofa(os.path.join(file_dir, files_2[1]))
        inverse_ctf.time = np.squeeze(inverse_ctf.time, 0)
        inverse_ctf = pf.dsp.resample(inverse_ctf, 48000, 'freq')

        brir = pf.dsp.convolve(brir, inverse_ctf, 'cut')

    # resample brir
    if sampling_rate != 48000:
        brir = pf.dsp.resample(brir, sampling_rate, post_filter=True)

    return brir


def headphone_impulse_responses(sampling_rate=44100):
    """
    Get Headphone Impulse Responses (HpIRs).

    The HpIRs are taken from the FABIAN database [#]_. They were measured with
    Sennheiser HD-650 headphones.

    .. note ::

        **License**: CC BY 4.0 	Fabian Brinkmann, Alexander Lindau, Stefan
        Weinzierl, Gunnar Geissler, Steven van de Par, Markus Müller-Trapet,
        Rob Opdam, Michael Vorländer

    Parameters
    ----------
    sampling_rate : int, optional
        The sampling rate of the HpIRs in Hz. The default of ``44100`` uses the
        HpIRs as they are, any other value uses :py:func:`~pyfar.dsp.resample`
        for resampling to the desired sampling rate.

    Returns
    -------
    hpirs : signal
        The HpIRs.

    References
    ----------
    .. [#] http://dx.doi.org/10.14279/depositonce-5718.5
    """

    # download files if requires
    files = _load_files('headphone_impulse_responses')

    # load HRIRs
    hpirs, *_ = pf.io.read_sofa(os.path.join(file_dir, files[0]))

    if sampling_rate != 44100:
        hpirs = pf.dsp.resample(hpirs, sampling_rate, 'freq', post_filter=True)

    return hpirs


def head_related_impulse_responses(
        position=[[0, 0]], diffuse_field_compensation=False,
        sampling_rate=44100):
    """
    Get HRIRs for specified source positions and sampling rate.

    The head-related impulse responses (HRIRs) are taken from the FABIAN
    database [#]_. They are shortened to 128 samples for convenience. HRIRs are
    available on the horizontal plane (elevation equals zero degrees, azimuth
    angles available between 0 and 360 degrees in steps of two degrees) and
    median plane (azimuth equals 0 or 180 degrees, elevation angles available
    between -90 and 90 degrees in steps of two degrees).

    .. note ::

        **License**: CC BY 4.0 	Fabian Brinkmann, Alexander Lindau, Stefan
        Weinzierl, Gunnar Geissler, Steven van de Par, Markus Müller-Trapet,
        Rob Opdam, Michael Vorländer

    Parameters
    ----------
    position : list, str, optional
        The positions for which HRIRs are returned.

        ``'horizontal'``
            Return horizontal plane HRIRs with an angular resolution of 2
            degrees.
        ``'median'``
            Return median plane HRIRs with an angular resolution of 2 degrees.
        Array like
            Return HRIRs at azimuth and elevation in degrees. For example
            ``[[30, 0], [330, 0]]`` returns HRIRs on the horizontal plane
            (0 degree elevation) for azimuth angles of 30 and 330 degrees.
        :py:class:`~pyfar.Coordinates`
            Return HRIRs at positions defined by a pyfar Coordinates object.
            Note that the HRIRs were measured at a fixed radius of 1.7 m.
            Hence, the radius in `position` will be ignored.

        The default is ``[[0, 0]]``, which returns the HRIR for frontal sound
        incidence. A ValueError is raised if the requested position is not
        available.
    diffuse_field_compensation : bool, optional
        Apply a diffuse field compensation to the HRIRs. This can be used as a
        simple headphone compensation filter when listening to the HRIRs. The
        default is False, which does not apply the compensation.
    sampling_rate : int, optional
        The sampling rate of the HRIRs in Hz. The default of ``44100`` uses the
        HRIRs as they are, any other value uses :py:func:`~pyfar.dsp.resample`
        for resampling to the desired sampling rate.

    Returns
    -------
    hrirs : signal
        The HRIRs.
    sources : Coordinates
        The source positions for which the HRIRs are returned.

    References
    ----------
    .. [#] http://dx.doi.org/10.14279/depositonce-5718.5
    """

    # download files if requires
    files = _load_files('head_related_impulse_responses')

    # load HRIRs
    hrirs, sources, _ = pf.io.read_sofa(os.path.join(file_dir, files[0]))

    # tolerance in radians for finding source positions. If this tolerance is
    # exceeded, an error is raised to inform the user
    tolerance_rad = 0.1 / 180 * np.pi

    # get indices of source positions
    if isinstance(position, str) and position == "horizontal":
        idx = sources.elevation == 0
    elif isinstance(position, str) and position == "median":
        idx = sources.lateral == 0
        idx = np.where(idx)[0]
        # sort positions according to polar angle
        polar = sources.polar[idx].flatten()
        idx = (idx[np.argsort(polar)], )
    else:

        if isinstance(position, pf.Coordinates):
            # force radius to be equal to those of HRIRs
            position.radius = 1.7
        else:
            # convert to pyfar coordinates for convenience
            position = np.atleast_2d(position) / 180 * np.pi
            position = pf.Coordinates.from_spherical_elevation(
                position[..., 0], position[..., 1], 1.7)

        # find requested HRIRs
        idx, distance = sources.find_nearest(
            position, distance_measure='spherical_radians')

        if np.any(distance > tolerance_rad):
            raise ValueError((
                "HRIRs for one or more requested source positions are not "
                "available. See documentation for available positions."))

    # select data for desired source positions
    hrirs.time = hrirs.time[idx]
    sources = sources[idx]

    # force cdim to be independent of number of requested source positions
    # must be done due to indexing and numpy broadcasting above
    if hrirs.cdim == 1:
        hrirs.time = hrirs.time[None, ...]

    # diffuse field compensation
    if diffuse_field_compensation:
        inverse_ctf, *_ = pf.io.read_sofa(os.path.join(file_dir, files[1]))

        hrirs = pf.dsp.convolve(hrirs, inverse_ctf, 'cut')
        hrirs.comment = (
            'Diffuse field compensated data from the FABIAN HRTF data base '
            'and shortened to 128 samples '
            '(http://dx.doi.org/10.14279/depositonce-5718.5)')
    else:
        hrirs.comment = (
            'Data from the FABIAN HRTF data base shortened to 128 samples '
            '(http://dx.doi.org/10.14279/depositonce-5718.5)')

    if sampling_rate != 44100:
        hrirs = pf.dsp.resample(hrirs, sampling_rate, 'freq', post_filter=True)

    return hrirs, sources


def room_impulse_response(sampling_rate=48000):
    """
    Get a room impulse response (RIR).

    The RIR was recorded with class I 1/2 inch measurement microphone in the
    Berliner Philharmonie [#]_ (Emitter 17). For more information see [#]_. A
    matching binaural room impulse response can be obtained by
    :py:func:`~brir`.

    .. note ::

        **License**: CC BY-NC-SA 4.0, David Ackermann, Audio Communication
        Group, Technical University of Berlin.

    Parameters
    ----------
    sampling_rate : int
        The sampling rate of the RIR in Hz. The default of ``48000`` uses the
        RIR as it is, any other value uses :py:func:`~pyfar.dsp.resample`
        for resampling to the desired sampling rate.

    Returns
    -------
    rir : Signal
        The RIR

    References
    ----------
    .. [#] http://dx.doi.org/10.14279/depositonce-15774

    .. [#] D. Ackermann, J. Domann, F. Brinkmann, J. M. Arend, M. Schneider,
           C. Pörschmann, and S. Weinzierl 'Recordings of a Loudspeaker
           Orchestra with Multi-Channel Microphone Arrays for the Evaluation of
           Spatial Audio Methods,' J. Audio Eng. Soc. (submitted)
    """

    # download files if requires
    files = _load_files('room_impulse_response')

    # load rir
    rir = pf.io.read_audio(os.path.join(file_dir, files[0]))

    # resample rir
    if sampling_rate != 48000:
        rir = pf.dsp.resample(rir, sampling_rate, post_filter=True)

    return rir


def _load_files(data):
    """Download files from Audio Communication Server if they do not exist."""

    # create directory if required
    if not os.path.isdir(file_dir):
        # provide verbose error for read-only file systems
        try:
            os.mkdir(file_dir)
        except OSError as error:
            if 'Read-only' in str(error):
                raise OSError((f'{data} can not be loaded because the file '
                               'system is read-only.')) from error
            else:
                raise error

    # set the filenames
    if data in ['binaural_room_impulse_response', 'castanets', 'drums',
                'guitar', 'room_impulse_response']:
        files = (f'{data}.wav', )
    elif data == 'headphone_impulse_responses':
        files = ('headphone_impulse_responses.sofa', )
    elif data == 'head_related_impulse_responses':
        files = ('head_related_impulse_responses.sofa',
                 'head_related_impulse_responses_ctf_inverted_smoothed.sofa',
                 'head_related_impulse_responses.py')
    elif data == 'speech':
        files = ('speech_female.wav', 'speech_male.wav')
    else:
        raise ValueError("Invalid data")

    files += (f"{data}_license.txt", )

    # check if files exist
    files_exist = True
    for file in files:
        if not os.path.isfile(os.path.join(file_dir, file)):
            files_exist = False
            break

    if files_exist:
        return files

    # download files
    print(f"Loading {data} data. This is only done once.")

    http = urllib3.PoolManager(cert_reqs='CERT_REQUIRED')
    url = 'https://pyfar.org/wp-content/uploads/pyfar_files/'

    for file in files:

        http_data = http.urlopen('GET', url + file)

        # save the data
        if http_data.status == 200:
            save_name = os.path.join(file_dir, file)
            with open(save_name, 'wb') as out:
                out.write(http_data.data)
        else:
            raise ConnectionError(
                "Connection error. Please check your internet connection.")

    return files

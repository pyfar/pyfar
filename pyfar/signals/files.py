import os
import numpy as np
import pyfar as pf


def brir(diffuse_field_compensation=False, sampling_rate=48000):
    """
    Get a binaural room impulse response (BRIR).

    The BRIR was recorded with the FABIAN head and torso simulator in the
    Berliner Philharmonie [#]_ (Emitter 17). The head of FABIAN was rotated
    25 degree to the right. For more information see [2]. A matching room
    impulse response can be obtained by :py:func:~`rir`.

    Parameters
    ----------
    diffuse_field_compensation : bool, optional
        Apply a diffuse field compensation to the BRIR. This can be used as a
        simple headphone compensation filter when listening to the BRIR. The
        default is False, which does not apply the compensation. The diffuse
        field compensation is taken from :py:func:`~hrirs`
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

    License
    -------
    CC BY-NC-SA 4.0, David Ackermann, Audio Communication Group, Technical
    University of Berlin
    """

    # download files if requires
    files = _load_files('brir')
    if diffuse_field_compensation:
        files_2 = _load_files('hrirs')

    # load brir
    brir = pf.io.read_audio(os.path.join(
        os.path.dirname(__file__), 'files', files[0]))

    # load and resample diffuse field filter
    if diffuse_field_compensation:
        inverse_ctf, *_ = pf.io.read_sofa(os.path.join(
            os.path.dirname(__file__), 'files', files_2[1]))
        inverse_ctf.time = np.squeeze(inverse_ctf.time, 0)
        inverse_ctf = pf.dsp.resample(inverse_ctf, 48000, 'freq')

        brir = pf.dsp.convolve(brir, inverse_ctf, 'cut')

    # resample brir
    if sampling_rate != 48000:
        brir = pf.dsp.resample(brir, sampling_rate)

    return brir


def castanets(sampling_rate=44100):
    """
    Get anechoic castanet sample.

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

    License
    -------
    CC 0, Matthias Frank (original source:
    https://iaem.at/Members/frank/sounds/)
    """

    # download files if requires
    files = _load_files('castanets')

    # load brir
    castanets = pf.io.read_audio(os.path.join(
        os.path.dirname(__file__), 'files', files[0]))

    castanets.fft_norm = "rms"

    # resample brir
    if sampling_rate != 44100:
        castanets = pf.dsp.resample(castanets, sampling_rate)

    return castanets


def drums(sampling_rate=44100):
    """
    Get dry drum sample.

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

    License
    -------
    CC BY Fabian Brinkmann, Johannes M. Arend
    """

    # download files if requires
    files = _load_files('drums')

    # load brir
    drums = pf.io.read_audio(os.path.join(
        os.path.dirname(__file__), 'files', files[0]))

    drums.fft_norm = "rms"

    # resample brir
    if sampling_rate != 48000:
        drums = pf.dsp.resample(drums, sampling_rate)

    return drums


def guitar(sampling_rate=48000):
    """
    Get anechoic guitar sample.

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

    License
    -------
    CC BY-SA Michio Woirgard, Philipp Stade, Jeffrey Amankwor, Benjamin
    Bernschütz, and Johannes Arend Audio Group, Cologne University of Applied
    Sciences (original data from
    http://audiogroup.web.th-koeln.de/anechoic.html, Flamenco2_U89.wav)
    """

    # download files if requires
    files = _load_files('guitar')

    # load brir
    guitar = pf.io.read_audio(os.path.join(
        os.path.dirname(__file__), 'files', files[0]))

    guitar.fft_norm = "rms"

    # resample brir
    if sampling_rate != 48000:
        guitar = pf.dsp.resample(guitar, sampling_rate)

    return guitar


def hrirs(position=[[0, 0]], diffuse_field_compensation=False,
          sampling_rate=44100):
    """
    Get HRIRs for specified source positions and sampling rate.

    The head-related impulse responses (HRIRs) are taken from the FABIAN
    database available from http://dx.doi.org/10.14279/depositonce-5718.5. They
    are shortened to 128 samples for convenience.

    Parameters
    ----------
    position : list, str, optional
        The positions for which HRIRs are returned. HRIRs are available on the
        horizontal and median plane in an angular resolution of 2 degrees.

        ``'horizontal'``
            Return horizontal plane HRIRs with an angular resolution of 2
            degrees.
        ``'median'``
            Return median plane HRIRs with an angular resolution of 2 degrees.
        List of coordinates
            Return HRIRs at specific positions defined by a list of azimuth
            and elevation values in degrees. For example
            ``[[30, 0], [330, 0]]`` returns HRIRs on the horizontal plane
            (0 degree elevation) for azimuth angles of 30 and 330 degrees.

        The default is ``[[0, 0]]``, which returns the HRIR for frontal sound
        incidence.
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
    source_positions : Coordinates
        The source positions for which the HRIRs are returned.

    License
    -------
    CC BY 4.0 	Fabian Brinkmann, Alexander Lindau, Stefan Weinzierl, Gunnar
    Geissler, Steven van de Par, Markus Müller-Trapet, Rob Opdam, Michael
    Vorländer (Original data from the FABIAN HRTF database,
    http://dx.doi.org/10.14279/depositonce-5718.5)
    """

    # download files if requires
    files = _load_files('hrirs')

    # load HRIRs
    hrirs, source_positions, _ = pf.io.read_sofa(
        os.path.join(os.path.dirname(__file__), 'files', files[0]))

    # get indices of source positions
    if position == "horizontal":
        _, mask = source_positions.find_slice('elevation', 'deg', 0)
    elif position == "median":
        _, mask = source_positions.find_slice('lateral', 'deg', 0)
    else:
        mask = np.full((358, ), False)
        for pos in position:
            _, mask_current = source_positions.find_nearest_sph(
                pos[0], pos[1], 1.7, distance=0,
                domain="sph", convention="top_elev", unit="deg")
            if np.any(mask_current):
                mask = mask | mask_current
            else:
                raise ValueError((
                    f"HRIR for azimuth={pos[0]} and elevation={pos[1]} degrees"
                    " is not available. See help for more information."))

    # select data for desired source positions
    hrirs.time = hrirs.time[mask]
    source_positions = source_positions[mask]

    # diffuse field compensation
    if diffuse_field_compensation:
        inverse_ctf, *_ = pf.io.read_sofa(os.path.join(
            os.path.dirname(__file__), 'files', files[1]))

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
        hrirs = pf.dsp.resample(hrirs, sampling_rate, 'freq')

    return hrirs, source_positions


def rir(sampling_rate=48000):
    """
    Get a room impulse response (RIR).

    The RRIR was recorded with class I 1/2 inch measurement microphpone in the
    Berliner Philharmonie [#]_ (Emitter 17). For more information see [2]. A
    matching binaural room impulse response can be obtained by
    :py:func:~`brir`.

    Parameters
    ----------
    sample_rate : int, optional
        The sampling rate of the BRIR in Hz. The default of ``48000`` uses the
        BRIR as it is, any other value uses :py:func:`~pyfar.dsp.resample`
        for resampling to the desired sampling rate.

    Returns
    -------
    rir : Signal
        The BRIR

    References
    ----------
    .. [#] http://dx.doi.org/10.14279/depositonce-15774

    .. [#] D. Ackermann, J. Domann, F. Brinkmann, J. M. Arend, M. Schneider,
           C. Pörschmann, and S. Weinzierl 'Recordings of a Loudspeaker
           Orchestra with Multi-Channel Microphone Arrays for the Evaluation of
           Spatial Audio Methods,' J. Audio Eng. Soc. (submitted)

    License
    -------
    CC BY-NC-SA 4.0, David Ackermann, Audio Communication Group, Technical
    University of Berlin
    """

    # download files if requires
    files = _load_files('rir')

    # load brir
    rir = pf.io.read_audio(os.path.join(
        os.path.dirname(__file__), 'files', files[0]))

    # resample brir
    if sampling_rate != 48000:
        rir = pf.dsp.resample(rir, sampling_rate)

    return rir


def speech(voice="female", sampling_rate=44100):
    """
    Get anechoic speech sample.

    The samples were taken from 'Music for Archimedes' [#]_ (Tracks 4, 5) with
    kind permission of Bang & Olufsen for research and personal purposes. Any
    commercial use or publication requires approval of Bang & Olufsen. For more
    information on the recordings see [#]_.

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

    License
    -------
    (c) Bang & Olufsen
    """

    # download files if requires
    files = _load_files('speech')

    # load brir
    file = files[0] if voice == "female" else files[1]
    speech = pf.io.read_audio(os.path.join(
        os.path.dirname(__file__), 'files', file))

    speech.fft_norm = "rms"

    # resample brir
    if sampling_rate != 44100:
        speech = pf.dsp.resample(speech, sampling_rate)

    return speech


def _load_files(data):

    if data in ['brir', 'castanets', 'drums', 'guitar', 'rir']:
        files = (f'{data}.wav', )
    elif data == 'hptf':
        files = ('hptf.sofa', )
    elif data == 'hrirs':
        files = ('hrirs.sofa', 'hrirs_ctf_inverted_smoothed.sofa', 'hrirs.py')
    elif data == 'speech':
        files = ('speech_female.wav', 'speech_male.wav')
    else:
        raise ValueError("Invalid data")

    files += (f"{files}_license.txt", )

    return files

import os
import numpy as np
import pyfar as pf


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
        to obtain HRIRs at the desired sampling rate.

    Returns
    -------
    hrirs : signal
        The HRIRs.
    source_positions : Coordinates
        The source positions for which the HRIRs are returned.
    """

    hrirs, source_positions, _ = pf.io.read_sofa(
        os.path.join(os.path.dirname(__file__), 'audio_files', 'hrirs.sofa'))

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
            os.path.dirname(__file__), 'audio_files',
            'hrirs_ctf_inverted_smoothed.sofa'))

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
        raise ValueError("tbd")

    return hrirs, source_positions

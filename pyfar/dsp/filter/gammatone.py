import numpy as np


def erb_frequencies(freq_range, resolution=1, reference_frequency=1000):
    """
    Get frequencies that are linearly spaced on the ERB frequency scale.

    The implementation follows Eq. (16) and (17) in [#]_ and was ported from
    the auditory modeling toolbox [#]_

    Parameters
    ----------
    freq_range : array like
        The upper and lower frequency limits in Hz between which the frequency
        vector is computed.
    resolution : number, optional
        The frequency resolution in ERB units. The default of ``1`` returns
        frequencies that are spaced by 1 ERB unit, a value of ``0.5`` would
        return frequencies that are spaced by 0.5 ERB units.
    reference_frequency : number, optional
        The reference frequency in Hz relative to which the frequency vector
        is constructed. The default is ``1000``.

    Returns
    -------
    frequencies : numpy array
        The frequencies in Hz that are linearly distributed on the ERB scale
        with a spacing given by `resolution` ERB units.

    References
    ----------
    .. [#] V. Hohmann, “Frequency analysis and synthesis using a gammatone
           filterbank,” Acta Acust. united Ac. 88, 433-442 (2002).

    .. [#] P. L. Søndergaard, and P. Majdak, “The auditory modeling toolbox,”
           in The technology of binaural listening, edited by J. Blauert
           (Heidelberg et al., Springer, 2013) pp. 33-56.


    """

    # check input
    if not isinstance(freq_range, (list, tuple, np.ndarray)) \
            or len(freq_range) != 2:
        raise ValueError("freq_range must be an array like of length 2")
    if freq_range[0] > freq_range[1]:
        raise ValueError(("The first value of freq_range must be smaller "
                          "than the second value"))
    if resolution <= 0:
        raise ValueError("Resolution must be larger than zero")

    # convert the frequency range and reference to ERB scale
    # (Hohmann 2002, Eq. 16)
    erb_range = 9.2645 * np.sign(freq_range) * np.log(
        1 + np.abs(freq_range) * 0.00437)
    erb_ref = 9.2645 * np.sign(reference_frequency) * np.log(
        1 + np.abs(reference_frequency) * 0.00437)

    # get the referenced range
    erb_ref_range = np.array([erb_ref - erb_range[0], erb_range[1] - erb_ref])

    # construct the frequencies on the ERB scale
    n_points = np.floor(erb_ref_range / resolution).astype(int)
    erb_points = np.arange(-n_points[0], n_points[1] + 1) * resolution \
        + erb_ref

    # convert to frequencies in Hz
    frequencies = 1 / 0.00437 * np.sign(erb_points) * (
        np.exp(np.abs(erb_points) / 9.2645) - 1)

    return frequencies

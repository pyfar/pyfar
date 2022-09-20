import pytest
import numpy as np
import numpy.testing as npt
import pyfar as pf


@pytest.mark.parametrize('function', [
    pf.signals.files.castanets,
    pf.signals.files.drums,
    pf.signals.files.guitar,
    pf.signals.files.binaural_room_impulse_response,
    pf.signals.files.room_impulse_response,
    pf.signals.files.headphone_impulse_responses
])
@pytest.mark.parametrize('sampling_rate', (44100, 48000))
def test_files(function, sampling_rate):
    """Test all files that only have the sampling rate as parameter"""

    # load data
    signal = function(sampling_rate=sampling_rate)
    # assert type and sampling rate
    assert isinstance(signal, pf.Signal)
    assert signal.sampling_rate == sampling_rate


@pytest.mark.parametrize('sampling_rate', (44100, 48000))
def test_speech(sampling_rate):

    # load data
    female = pf.signals.files.speech('female', sampling_rate)
    male = pf.signals.files.speech('male', sampling_rate)
    # assert type and sampling rate
    assert isinstance(female, pf.Signal)
    assert isinstance(male, pf.Signal)

    assert female.sampling_rate == sampling_rate
    assert male.sampling_rate == sampling_rate

    # pad to same length
    female = pf.dsp.pad_zeros(
        female, max(male.n_samples, female.n_samples)-female.n_samples)
    male = pf.dsp.pad_zeros(
        male, max(male.n_samples, female.n_samples)-male.n_samples)
    # normalize
    female.time /= np.max(np.abs(female.time))
    male.time /= np.max(np.abs(male.time))

    # compare energy up to 100 Hz (male voice must contain more energy)
    f_id = female.find_nearest_frequency(100)
    assert np.sum(np.abs(male.freq_raw[:f_id])) > \
        np.sum(np.abs(female.freq_raw[:f_id]))


@pytest.mark.parametrize('position,convention,first,second', [
    ([[0, 20]], 'top_elev', [0], [20]),
    ([[0, 20], [10, 0]], 'top_elev', [0, 10], [0, 20]),
    ('horizontal', 'top_elev', np.arange(0, 180)*2, np.zeros(180)),
    ('median', 'side', np.zeros(180), np.arange(-45, 135)*2)])
def test_hrirs_position(position, convention, first, second):
    """Test `position` argument"""

    # get the data
    hrirs, sources = pf.signals.files.head_related_impulse_responses(position)

    # test hrirs
    assert hrirs.cshape == (len(first), 2)

    # test source positions
    sg = sources.get_sph(convention, 'deg')
    npt.assert_allclose(first, np.sort(sg[..., 0].flatten()), atol=1e-12)
    npt.assert_allclose(second, np.sort(sg[..., 1].flatten()), atol=1e-12)


def test_hrirs_diffuse_field_compensation():
    """Test diffuse field compensation for HRIRs"""
    hrirs, _ = pf.signals.files.head_related_impulse_responses(
        diffuse_field_compensation=False)
    dtfs, _ = pf.signals.files.head_related_impulse_responses(
        diffuse_field_compensation=True)

    # HRIRs must contain more energy
    assert np.all(np.sum(hrirs.time**2, -1) - np.sum(dtfs.time**2, -1) > 0)

    # Check comments
    assert hrirs.comment.startswith("Data from")
    assert dtfs.comment.startswith("Diffuse field compensated data from")


@pytest.mark.parametrize('sampling_rate', (44100, 48000))
def test_hrirs_sampling_rate(sampling_rate):
    """Test getting HRIRs in different sampling rates"""

    hrirs, _ = pf.signals.files.head_related_impulse_responses(
        sampling_rate=sampling_rate)
    assert hrirs.sampling_rate == sampling_rate


def test_hrirs_assertions():
    """Test assertions for getting HRIRs"""
    with pytest.raises(ValueError, match="HRIR for azimuth=1"):
        pf.signals.files.head_related_impulse_responses([[1, 0]])

import pytest
import numpy as np
import numpy.testing as npt
import pyfar as pf


@pytest.mark.parametrize('position,convention,first,second', [
    ([[0, 20]], 'top_elev', [0], [20]),
    ([[0, 20], [10, 0]], 'top_elev', [0, 10], [0, 20]),
    ('horizontal', 'top_elev', np.arange(0, 180)*2, np.zeros(180)),
    ('median', 'side', np.zeros(180), np.arange(-45, 135)*2)])
def test_hrirs_position(position, convention, first, second):
    """Test `position` argument"""

    # get the data
    hrirs, sources = pf.signals.hrirs(position)

    # test hrirs
    assert hrirs.cshape == (len(first), 2)

    # test source positions
    sg = sources.get_sph(convention, 'deg')
    npt.assert_allclose(first, np.sort(sg[..., 0].flatten()), atol=1e-12)
    npt.assert_allclose(second, np.sort(sg[..., 1].flatten()), atol=1e-12)


def test_hrirs_diffuse_field_compensation():
    """Test diffuse field compenstation for HRIRs"""
    hrirs, _ = pf.signals.hrirs(diffuse_field_compensation=False)
    dtfs, _ = pf.signals.hrirs(diffuse_field_compensation=True)

    # HRIRs must contain more energy
    assert np.all(np.sum(hrirs.time**2, -1) - np.sum(dtfs.time**2, -1) > 0)

    # Check comments
    assert hrirs.comment.startswith("Data from")
    assert dtfs.comment.startswith("Diffuse field compensated data from")


def test_hrirs_sampling_rate():
    """Test geting HRIRs in different sampling rates"""
    pass


def test_hrirs_assertions():
    """Test assertions for getting HRIRs"""
    with pytest.raises(ValueError, match="HRIR for azimuth=1"):
        pf.signals.hrirs([[1, 0]])

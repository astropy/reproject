# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os

import numpy as np
import pytest
from astropy import units as u
from astropy.wcs import WCS
from astropy.wcs.wcsapi import HighLevelWCSWrapper, SlicedLowLevelWCS
from numpy.testing import assert_allclose

from ..high_level import reproject_adaptive
from ...tests.helpers import array_footprint_to_hdulist

DATA = os.path.join(os.path.dirname(__file__), '..', '..', 'tests', 'data')


def as_high_level_wcs(wcs):
    return HighLevelWCSWrapper(SlicedLowLevelWCS(wcs, Ellipsis))


@pytest.mark.array_compare(single_reference=True)
@pytest.mark.parametrize('wcsapi', (False, True))
def test_reproject_adaptive_2d(wcsapi):

    # Set up initial array with pattern
    data_in = np.zeros((256, 256))
    data_in[::20, :] = 1
    data_in[:, ::20] = 1
    data_in[10::20, 10::20] = 1

    # Define a simple input WCS
    wcs_in = WCS(naxis=2)
    wcs_in.wcs.crpix = 128.5, 128.5
    wcs_in.wcs.cdelt = -0.01, 0.01

    # Define a lower resolution output WCS with rotation
    wcs_out = WCS(naxis=2)
    wcs_out.wcs.crpix = 30.5, 30.5
    wcs_out.wcs.cdelt = -0.0427, 0.0427

    header_out = wcs_out.to_header()

    if wcsapi:  # Enforce a pure wcsapi API
        wcs_in = as_high_level_wcs(wcs_in)
        wcs_out = as_high_level_wcs(wcs_out)

    array_out, footprint_out = reproject_adaptive((data_in, wcs_in),
                                                  wcs_out, shape_out=(60, 60))

    # Check that surface brightness is conserved in the unrotated case
    assert_allclose(np.nansum(data_in), np.nansum(array_out) * (256 / 60) ** 2, rtol=0.1)

    # ASTROPY_LT_40: astropy v4.0 introduced new default header keywords,
    # once we support only astropy 4.0 and later we can update the reference
    # data files and remove this section.
    for key in ('DATEREF', 'MJDREFF', 'MJDREFI'):
        header_out.pop(key, None)

    return array_footprint_to_hdulist(array_out, footprint_out, header_out)


@pytest.mark.array_compare(single_reference=True)
def test_reproject_adaptive_2d_rotated():

    # Set up initial array with pattern
    data_in = np.zeros((256, 256))
    data_in[::20, :] = 1
    data_in[:, ::20] = 1
    data_in[10::20, 10::20] = 1

    # Define a simple input WCS
    wcs_in = WCS(naxis=2)
    wcs_in.wcs.crpix = 128.5, 128.5
    wcs_in.wcs.cdelt = -0.01, 0.01

    # Define a lower resolution output WCS with rotation
    wcs_out = WCS(naxis=2)
    wcs_out.wcs.crpix = 30.5, 30.5
    wcs_out.wcs.cdelt = -0.0427, 0.0427
    wcs_out.wcs.pc = [[0.8, 0.2], [-0.2, 0.8]]

    header_out = wcs_out.to_header()

    array_out, footprint_out = reproject_adaptive((data_in, wcs_in),
                                                  wcs_out, shape_out=(60, 60))

    # ASTROPY_LT_40: astropy v4.0 introduced new default header keywords,
    # once we support only astropy 4.0 and later we can update the reference
    # data files and remove this section.
    for key in ('DATEREF', 'MJDREFF', 'MJDREFI'):
        header_out.pop(key, None)

    return array_footprint_to_hdulist(array_out, footprint_out, header_out)


@pytest.mark.array_compare(single_reference=True)
@pytest.mark.parametrize('file_format', ['fits', 'asdf'])
def test_reproject_adaptive_roundtrip(file_format):

    # Test the reprojection with solar data, which ensures that the masking of
    # pixels based on round-tripping works correctly. Using asdf is not just
    # about testing a different format but making sure that GWCS works.

    pytest.importorskip('sunpy', minversion='1.0.4')
    from sunpy.map import Map
    from sunpy.coordinates.ephemeris import get_body_heliographic_stonyhurst

    if file_format == 'fits':
        map_aia = Map(os.path.join(DATA, 'aia_171_level1.fits'))
        data = map_aia.data
        wcs = map_aia.wcs
        date = map_aia.date
        target_wcs = wcs.deepcopy()
    elif file_format == 'asdf':
        pytest.importorskip('astropy', minversion='4.0')
        pytest.importorskip('gwcs', minversion='0.12')
        asdf = pytest.importorskip('asdf')
        aia = asdf.open(os.path.join(DATA, 'aia_171_level1.asdf'))
        data = aia['data'][...]
        wcs = aia['wcs']
        date = wcs.output_frame.reference_frame.obstime
        target_wcs = Map(os.path.join(DATA, 'aia_171_level1.fits')).wcs.deepcopy()
    else:
        raise ValueError('file_format should be fits or asdf')

    # Reproject to an observer on Venus

    target_wcs.wcs.cdelt = ([24, 24]*u.arcsec).to(u.deg)
    target_wcs.wcs.crpix = [64, 64]
    venus = get_body_heliographic_stonyhurst('venus', date)
    target_wcs.heliographic_observer = venus

    output, footprint = reproject_adaptive((data, wcs), target_wcs, (128, 128))

    header_out = target_wcs.to_header()

    # ASTROPY_LT_40: astropy v4.0 introduced new default header keywords,
    # once we support only astropy 4.0 and later we can update the reference
    # data files and remove this section.
    for key in ('CRLN_OBS', 'CRLT_OBS', 'DSUN_OBS', 'HGLN_OBS', 'HGLT_OBS',
                'MJDREFF', 'MJDREFI', 'RSUN_REF'):
        header_out.pop(key, None)
    header_out['DATE-OBS'] = header_out['DATE-OBS'].replace('T', ' ')

    return array_footprint_to_hdulist(output, footprint, header_out)

# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import itertools

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS
from astropy_healpix import nside_to_npix

from ...interpolation.tests.test_core import as_high_level_wcs
from ...tests.test_high_level import ALL_DTYPES
from ..high_level import reproject_from_healpix, reproject_to_healpix

DATA = os.path.join(os.path.dirname(__file__), 'data')


def get_reference_header(oversample=2, nside=1):

    reference_header = fits.Header()
    reference_header.update({
        'CDELT1': -180.0 / (oversample * 4 * nside),
        'CDELT2': 180.0 / (oversample * 4 * nside),
        'CRPIX1': oversample * 4 * nside,
        'CRPIX2': oversample * 2 * nside,
        'CRVAL1': 180.0,
        'CRVAL2': 0.0,
        'CTYPE1': 'RA---CAR',
        'CTYPE2': 'DEC--CAR',
        'CUNIT1': 'deg',
        'CUNIT2': 'deg',
        'NAXIS': 2,
        'NAXIS1': oversample * 8 * nside,
        'NAXIS2': oversample * 4 * nside})

    return reference_header


@pytest.mark.parametrize("wcsapi,nside,nested,healpix_system,image_system,dtype",
                         itertools.product([True, False], [1, 2, 4, 8, 16, 32, 64],
                                           [True, False], 'C', 'C', ALL_DTYPES))
def test_reproject_healpix_to_image_round_trip(wcsapi, nside, nested,
                                               healpix_system, image_system, dtype):
    """Test round-trip HEALPix->WCS->HEALPix conversion for a random map,
    with a WCS projection large enough to store each HEALPix pixel"""

    npix = nside_to_npix(nside)
    healpix_data = np.random.uniform(size=npix).astype(dtype)

    reference_header = get_reference_header(oversample=2, nside=nside)

    wcs_out = WCS(reference_header)
    shape_out = reference_header['NAXIS2'], reference_header['NAXIS1']

    if wcsapi:
        wcs_out = as_high_level_wcs(wcs_out)

    image_data, footprint = reproject_from_healpix(
        (healpix_data, healpix_system), wcs_out, shape_out=shape_out,
        order='nearest-neighbor', nested=nested)

    healpix_data_2, footprint = reproject_to_healpix(
        (image_data, wcs_out), healpix_system,
        nside=nside, order='nearest-neighbor', nested=nested)

    np.testing.assert_array_equal(healpix_data, healpix_data_2)


def test_reproject_file():
    reference_header = get_reference_header(oversample=2, nside=8)
    data, footprint = reproject_from_healpix(os.path.join(DATA, 'bayestar.fits.gz'),
                                             reference_header)
    reference_result = fits.getdata(os.path.join(DATA, 'reference_result.fits'))
    np.testing.assert_allclose(data, reference_result, rtol=1.e-5)


def test_reproject_invalid_order():
    reference_header = get_reference_header(oversample=2, nside=8)
    with pytest.raises(ValueError) as exc:
        reproject_from_healpix(os.path.join(DATA, 'bayestar.fits.gz'),
                               reference_header, order='bicubic')
    assert exc.value.args[0] == "Only nearest-neighbor and bilinear interpolation are supported"

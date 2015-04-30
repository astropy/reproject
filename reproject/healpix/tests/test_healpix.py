# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import itertools

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import pytest

from ..core import healpix_to_image, image_to_healpix


@pytest.mark.importorskip('healpy')
@pytest.mark.parametrize("nside,nest,healpix_system,image_system",
    itertools.product([1, 2, 4, 8, 16, 32, 64], [True, False], 'C', 'C'))
def test_reproject_healpix_to_image_round_trip(
        nside, nest, healpix_system, image_system):
    """Test round-trip HEALPix->WCS->HEALPix conversion for a random map,
    with a WCS projection large enough to store each HEALPix pixel"""
    import healpy as hp

    npix = hp.nside2npix(nside)
    healpix_data = np.random.uniform(size=npix)

    oversample = 2
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

    wcs_out = WCS(reference_header)
    shape_out = reference_header['NAXIS2'], reference_header['NAXIS1']

    image_data = healpix_to_image(
        healpix_data, healpix_system, wcs_out, shape_out,
        interp=False, nest=nest)

    healpix_data_2 = image_to_healpix(
        image_data, wcs_out, healpix_system,
        nside, interp=False, nest=nest)

    np.testing.assert_array_equal(healpix_data, healpix_data_2)

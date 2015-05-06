# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import itertools

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.utils.data import get_pkg_data_filename
from astropy.tests.helper import pytest

from .. import reproject_interp, reproject_exact

# TODO: add reference comparisons

ALL_MODES = ('nearest-neighbor',
             'bilinear',
             'biquadratic',
             'bicubic',
             'flux-conserving')

ALL_DTYPES = []
for endian in ('<', '>'):
    for kind in ('u', 'i', 'f'):
        for size in ('1', '2', '4', '8'):
            if kind == 'f' and size == '1':
                continue
            ALL_DTYPES.append(np.dtype(endian + kind + size))


class TestReproject(object):

    def setup_method(self, method):

        self.header_in = fits.Header.fromtextfile(get_pkg_data_filename('data/gc_ga.hdr'))
        self.header_out = fits.Header.fromtextfile(get_pkg_data_filename('data/gc_eq.hdr'))

        self.header_out_size = self.header_out.copy()
        self.header_out_size['NAXIS'] = 2
        self.header_out_size['NAXIS1'] = 600
        self.header_out_size['NAXIS2'] = 550

        self.array_in = np.ones((700, 690))

        self.hdu_in = fits.ImageHDU(self.array_in, self.header_in)

        self.wcs_in = WCS(self.header_in)
        self.wcs_out = WCS(self.header_out)
        self.shape_out = (600, 550)

    def test_hdu_header(self):

        with pytest.raises(ValueError) as exc:
            reproject_interp(self.hdu_in, self.header_out)
        assert exc.value.args[0] == "Need to specify shape since output header does not contain complete shape information"

        reproject_interp(self.hdu_in, self.header_out_size)

    def test_hdu_wcs(self):
        reproject_interp(self.hdu_in, self.wcs_out, shape_out=self.shape_out)

    def test_array_wcs_header(self):

        with pytest.raises(ValueError) as exc:
            reproject_interp((self.array_in, self.wcs_in), self.header_out)
        assert exc.value.args[0] == "Need to specify shape since output header does not contain complete shape information"

        reproject_interp((self.array_in, self.wcs_in), self.header_out_size)

    def test_array_wcs_wcs(self):
        reproject_interp((self.array_in, self.wcs_in), self.wcs_out, shape_out=self.shape_out)

    def test_array_header_header(self):
        reproject_interp((self.array_in, self.header_in), self.header_out_size)


INPUT_HDR = """
WCSAXES =                    2 / Number of coordinate axes
CRPIX1  =                  0.5 / Pixel coordinate of reference point
CRPIX2  =                  0.5 / Pixel coordinate of reference point
CDELT1  =         -0.001666666 / [deg] Coordinate increment at reference point
CDELT2  =          0.001666666 / [deg] Coordinate increment at reference point
CUNIT1  = 'deg'                / Units of coordinate increment and value
CUNIT2  = 'deg'                / Units of coordinate increment and value
CTYPE1  = 'GLON-CAR'           / galactic longitude, plate caree projection
CTYPE2  = 'GLAT-CAR'           / galactic latitude, plate caree projection
CRVAL1  =                  0.0 / [deg] Coordinate value at reference point
CRVAL2  =                  0.0 / [deg] Coordinate value at reference point
LONPOLE =                  0.0 / [deg] Native longitude of celestial pole
LATPOLE =                 90.0 / [deg] Native latitude of celestial pole
"""


@pytest.mark.parametrize('projection_type, dtype', itertools.product(ALL_MODES, ALL_DTYPES))
def test_surface_brightness(projection_type, dtype):

    header_in = fits.Header.fromstring(INPUT_HDR, sep='\n')
    header_in['NAXIS'] = 2
    header_in['NAXIS1'] = 10
    header_in['NAXIS2'] = 10

    header_out = header_in.copy()

    header_out['CDELT1'] /= 2
    header_out['CDELT2'] /= 2
    header_out['NAXIS1'] *= 2
    header_out['NAXIS2'] *= 2

    data_in = np.ones((10, 10), dtype=dtype)

    if projection_type == 'flux-conserving':
        data_out, footprint = reproject_exact((data_in, header_in), header_out)
    else:
        data_out, footprint = reproject_interp((data_in, header_in), header_out,
                                               order=projection_type)

    assert data_out.shape == (20, 20)

    # Here we check that the values are still 1 despite the change in
    # resolution, which demonstrates that we are preserving surface
    # brightness.
    np.testing.assert_allclose(data_out, 1)

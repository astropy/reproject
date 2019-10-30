# Licensed under a 3-clause BSD style license - see LICENSE.rst

import itertools

import numpy as np
import pytest
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from astropy.wcs import WCS

from .. import reproject_exact, reproject_interp, reproject_adaptive

# TODO: add reference comparisons

ALL_MODES = ('nearest-neighbor',
             'bilinear',
             'biquadratic',
             'bicubic',
             'flux-conserving',
             'adaptive-nearest-neighbor',
             'adaptive-bilinear')

ALL_DTYPES = []
for endian in ('<', '>'):
    for kind in ('u', 'i', 'f'):
        for size in ('1', '2', '4', '8'):
            if kind == 'f' and size == '1':
                continue
            ALL_DTYPES.append(np.dtype(endian + kind + size))


@pytest.fixture(params=[reproject_interp, reproject_adaptive, reproject_exact],
                ids=["interp", "adaptive", "exact"])
def reproject_function(request):
    return request.param


class TestReproject:

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

    def test_hdu_header(self, reproject_function):

        with pytest.raises(ValueError) as exc:
            reproject_function(self.hdu_in, self.header_out)
        assert exc.value.args[0] == ("Need to specify shape since output header "
                                     "does not contain complete shape information")

        reproject_interp(self.hdu_in, self.header_out_size)

    def test_hdu_wcs(self, reproject_function):
        reproject_function(self.hdu_in, self.wcs_out, shape_out=self.shape_out)

    def test_array_wcs_header(self, reproject_function):

        with pytest.raises(ValueError) as exc:
            reproject_function((self.array_in, self.wcs_in), self.header_out)
        assert exc.value.args[0] == ("Need to specify shape since output header "
                                     "does not contain complete shape information")

        reproject_function((self.array_in, self.wcs_in), self.header_out_size)

    def test_array_wcs_wcs(self, reproject_function):
        reproject_function((self.array_in, self.wcs_in), self.wcs_out, shape_out=self.shape_out)

    def test_array_header_header(self, reproject_function):
        reproject_function((self.array_in, self.header_in), self.header_out_size)

    def test_return_footprint(self, reproject_function):
        array = reproject_function(self.hdu_in, self.wcs_out,
                                   shape_out=self.shape_out, return_footprint=False)
        assert isinstance(array, np.ndarray)


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
    elif projection_type.startswith('adaptive'):
        data_out, footprint = reproject_adaptive((data_in, header_in), header_out,
                                                 order=projection_type.split('-', 1)[1])
    else:
        data_out, footprint = reproject_interp((data_in, header_in), header_out,
                                               order=projection_type)

    assert data_out.shape == (20, 20)

    # Here we check that the values are still 1 despite the change in
    # resolution, which demonstrates that we are preserving surface
    # brightness.
    np.testing.assert_allclose(data_out, 1)


IDENTITY_TEST_HDR = """
NAXIS   =                    2 / Number of coordinate axes
NAXIS1  =                    5
NAXIS2  =                   10
CRPIX1  =  8540.80750619681    / Pixel coordinate of reference point
CRPIX2  =   4108.61481031444   / Pixel coordinate of reference point
CUNIT1  = 'deg'                / Units of coordinate increment and value
CUNIT2  = 'deg'                / Units of coordinate increment and value
CTYPE1  = 'RA---TAN'           / galactic longitude, plate caree projection
CTYPE2  = 'DEC--TAN'           / galactic latitude, plate caree projection
CRVAL1  =  282.582253365684    / [deg] Coordinate value at reference point
CRVAL2  =   -5.80644283270032  / [deg] Coordinate value at reference point
CD1_1   = 5.92448829959494E-05 / WCS transform matrix element
CD2_1   = -2.5640855053008E-08 / WCS transform matrix element
CD1_2   = 1.30443859769625E-08 / WCS transform matrix element
CD2_2   = 5.92479929826863E-05 / WCS transform matrix element
"""


@pytest.mark.parametrize('projection_type', ALL_MODES)
def test_identity_projection(projection_type):
    """Sanity check: identical input & output headers should preserve image."""
    header_in = fits.Header.fromstring(IDENTITY_TEST_HDR, sep='\n')
    data_in = np.random.rand(header_in['NAXIS2'], header_in['NAXIS1'])
    if projection_type == 'flux-conserving':
        data_out, footprint = reproject_exact((data_in, header_in), header_in)
    elif projection_type.startswith('adaptive'):
        data_out, footprint = reproject_adaptive((data_in, header_in), header_in,
                                                 order=projection_type.split('-', 1)[1])
    else:
        data_out, footprint = reproject_interp((data_in, header_in), header_in,
                                               order=projection_type)
    # When reprojecting with an identical input and output header,
    # we may expect the input and output data to be similar,
    # and the footprint values to be ~ones.
    expected_footprint = np.ones((header_in['NAXIS2'], header_in['NAXIS1']))
    np.testing.assert_allclose(footprint, expected_footprint)
    np.testing.assert_allclose(data_in, data_out, rtol=1e-6)

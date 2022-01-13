# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import random

import numpy as np
import pytest
from astropy.wcs import WCS
from astropy.io import fits
from astropy.io.fits import Header

from numpy.testing import assert_allclose

from reproject import reproject_exact, reproject_interp
from reproject.mosaicking.coadd import reproject_and_coadd
from reproject.tests.helpers import array_footprint_to_hdulist

ATOL = 1.e-9

DATA = os.path.join(os.path.dirname(__file__), '..', '..', 'tests', 'data')


@pytest.fixture(params=[reproject_interp, reproject_exact],
                ids=["interp", "exact"])
def reproject_function(request):
    return request.param


class TestReprojectAndCoAdd():

    def setup_method(self, method):

        self.wcs = WCS(naxis=2)
        self.wcs.wcs.ctype = 'RA---TAN', 'DEC--TAN'
        self.wcs.wcs.crpix = 322, 151
        self.wcs.wcs.crval = 43, 23
        self.wcs.wcs.cdelt = -0.1, 0.1
        self.wcs.wcs.equinox = 2000.

        self.array = np.random.random((399, 334))

    def _get_tiles(self, views):

        # Given a list of views as (imin, imax, jmin, jmax), construct
        #  tiles that can be passed into the co-adding code

        input_data = []

        for (jmin, jmax, imin, imax) in views:
            array = self.array[jmin:jmax, imin:imax].copy()
            wcs = self.wcs.deepcopy()
            wcs.wcs.crpix[0] -= imin
            wcs.wcs.crpix[1] -= jmin
            input_data.append((array, wcs))

        return input_data

    @property
    def _nonoverlapping_views(self):

        ie = (0, 122, 233, 245, 334)
        je = (0, 44, 45, 333, 335, 399)

        views = []
        for i in range(4):
            for j in range(5):
                views.append((je[j], je[j+1], ie[i], ie[i+1]))

        return views

    @property
    def _overlapping_views(self):

        ie = (0, 122, 233, 245, 334)
        je = (0, 44, 45, 333, 335, 399)

        views = []
        for i in range(4):
            for j in range(5):
                views.append((je[j], je[j+1] + 10, ie[i], ie[i+1] + 10))

        return views

    @pytest.mark.parametrize('combine_function', ['mean', 'sum'])
    def test_coadd_no_overlap(self, combine_function, reproject_function):

        # Make sure that if all tiles are exactly non-overlapping, and
        # we use 'sum' or 'mean', we get the exact input array back.

        input_data = self._get_tiles(self._nonoverlapping_views)

        input_data = [(self.array, self.wcs)]
        array, footprint = reproject_and_coadd(input_data, self.wcs,
                                               shape_out=self.array.shape,
                                               combine_function=combine_function,
                                               reproject_function=reproject_function)

        assert_allclose(array, self.array, atol=ATOL)
        assert_allclose(footprint, 1, atol=ATOL)

    def test_coadd_with_overlap(self, reproject_function):

        # Here we make the input tiles overlapping. We can only check the
        # mean, not the sum.

        input_data = self._get_tiles(self._overlapping_views)

        array, footprint = reproject_and_coadd(input_data, self.wcs,
                                               shape_out=self.array.shape,
                                               combine_function='mean',
                                               reproject_function=reproject_function)

        assert_allclose(array, self.array, atol=ATOL)

    def test_coadd_background_matching(self, reproject_function):

        # Test out the background matching

        input_data = self._get_tiles(self._overlapping_views)

        for array, wcs in input_data:
            array += random.uniform(-3, 3)

        # First check that without background matching the arrays don't match

        array, footprint = reproject_and_coadd(input_data, self.wcs,
                                               shape_out=self.array.shape,
                                               combine_function='mean',
                                               reproject_function=reproject_function)

        assert not np.allclose(array, self.array, atol=ATOL)

        # Now check that once the backgrounds are matched the values agree

        array, footprint = reproject_and_coadd(input_data, self.wcs,
                                               shape_out=self.array.shape,
                                               combine_function='mean',
                                               reproject_function=reproject_function,
                                               match_background=True)

        # The absolute values of the two arrays will be offset since any
        # solution that reproduces the offsets between images is valid

        assert_allclose(array - np.mean(array),
                        self.array - np.mean(self.array), atol=ATOL)

    def test_coadd_background_matching_with_nan(self, reproject_function):

        # Test out the background matching when NaN values are present. We do
        # this by using three arrays with the same footprint but with different
        # parts masked.

        array1 = self.array.copy() + random.uniform(-3, 3)
        array2 = self.array.copy() + random.uniform(-3, 3)
        array3 = self.array.copy() + random.uniform(-3, 3)

        array1[:, 122:] = np.nan
        array2[:, :50] = np.nan
        array2[:, 266:] = np.nan
        array3[:, :199] = np.nan

        input_data = [(array1, self.wcs), (array2, self.wcs), (array3, self.wcs)]

        array, footprint = reproject_and_coadd(input_data, self.wcs,
                                               shape_out=self.array.shape,
                                               combine_function='mean',
                                               reproject_function=reproject_function,
                                               match_background=True)

        # The absolute values of the two arrays will be offset since any
        # solution that reproduces the offsets between images is valid

        assert_allclose(array - np.mean(array),
                        self.array - np.mean(self.array), atol=ATOL)

    @pytest.mark.filterwarnings('ignore:unclosed file:ResourceWarning')
    @pytest.mark.parametrize('mode', ['arrays', 'filenames', 'hdus'])
    def test_coadd_with_weights(self, tmpdir, reproject_function, mode):

        # Make sure that things work properly when specifying weights

        array1 = self.array + 1
        array2 = self.array - 1

        weight1 = np.cumsum(np.ones_like(self.array), axis=1) - 1
        weight2 = weight1[:, ::-1]

        input_data = [(array1, self.wcs), (array2, self.wcs)]

        if mode == 'arrays':
            input_weights = [weight1, weight2]
        elif mode == 'filenames':
            filename1 = tmpdir.join('weight1.fits').strpath
            filename2 = tmpdir.join('weight2.fits').strpath
            fits.writeto(filename1, weight1)
            fits.writeto(filename2, weight2)
            input_weights = [filename1, filename2]
        elif mode == 'hdus':
            hdu1 = fits.ImageHDU(weight1)
            hdu2 = fits.ImageHDU(weight2)
            input_weights = [hdu1, hdu2]

        array, footprint = reproject_and_coadd(input_data, self.wcs,
                                               shape_out=self.array.shape,
                                               combine_function='mean',
                                               input_weights=input_weights,
                                               reproject_function=reproject_function,
                                               match_background=False)

        expected = self.array + (2 * (weight1 / weight1.max()) - 1)

        assert_allclose(array, expected, atol=ATOL)


HEADER_SOLAR_OUT = """
WCSAXES =                    2
CRPIX1  =                 90.5
CRPIX2  =                 45.5
CDELT1  =                    2
CDELT2  =                    2
CUNIT1  = 'deg'
CUNIT2  = 'deg'
CTYPE1  = 'HGLN-CAR'
CTYPE2  = 'HGLT-CAR'
CRVAL1  =                  0.0
CRVAL2  =                  0.0
LONPOLE =                  0.0
LATPOLE =                 90.0
DATE-OBS= '2011-02-15T00:14:03.654'
MJD-OBS =      55607.009764514
MJD-OBS =      55607.009764514
"""


@pytest.mark.array_compare()
def test_coadd_solar_map():

    # This is a test that exercises a lot of different parts of the mosaicking
    # code. The idea is to take three solar images from different viewpoints
    # and combine them into a single one. This uses weight maps that are not
    # uniform and also include NaN values.

    # The reference image was generated for sunpy 3.0.1 - it will not work with
    # previous versions due to the bug that https://github.com/sunpy/sunpy/pull/5381
    # fixes.
    pytest.importorskip('sunpy', minversion='3.0.1')
    from sunpy.map import Map, all_coordinates_from_map

    # Load in three images from different viewpoints around the Sun
    filenames = ['secchi_l0_a.fits', 'aia_171_level1.fits', 'secchi_l0_b.fits']
    maps = [Map(os.path.join(DATA, f)) for f in filenames]

    # Produce weight maps that are centered on the solar disk and go to zero at the edges
    coordinates = tuple(map(all_coordinates_from_map, maps))
    input_weights = [coord.transform_to("heliocentric").z.value for coord in coordinates]
    input_weights = [(w / np.nanmax(w)) ** 4 for w in input_weights]

    shape_out = [90, 180]
    wcs_out = WCS(Header.fromstring(HEADER_SOLAR_OUT, sep='\n'))
    scales = [1/6, 1, 1/6]

    input_data = tuple((a.data * scale, a.wcs) for (a, scale) in zip(maps, scales))

    array, footprint = reproject_and_coadd(input_data, wcs_out, shape_out,
                                           input_weights=input_weights,
                                           reproject_function=reproject_interp,
                                           match_background=True)

    header_out = wcs_out.to_header()

    # ASTROPY_LT_40: astropy v4.0 introduced new default header keywords,
    # once we support only astropy 4.0 and later we can update the reference
    # data files and remove this section.
    for key in ('MJDREFF', 'MJDREFI', 'MJDREF', 'MJD-OBS'):
        header_out.pop(key, None)

    return array_footprint_to_hdulist(array, footprint, header_out)

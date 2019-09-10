# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import absolute_import, division, print_function

import random

import pytest

import numpy as np
from numpy.testing import assert_allclose, assert_equal

from astropy.wcs import WCS

from ... import reproject_interp, reproject_exact

from ..coadd import reproject_and_coadd


ATOL = 1.e-9

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

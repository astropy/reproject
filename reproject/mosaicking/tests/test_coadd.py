# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import absolute_import, division, print_function

import random

import pytest

import numpy as np
from numpy.testing import assert_allclose, assert_equal

from astropy.wcs import WCS

try:
    import shapely  # noqa
except ImportError:
    SHAPELY_INSTALLED = False
else:
    SHAPELY_INSTALLED = True

from ... import reproject_interp

from ..coadd import reproject_and_coadd

class TestReprojectAndCoAdd():

    def setup_method(self, method):

        self.wcs = WCS(naxis=2)
        self.wcs.wcs.ctype = 'RA---TAN', 'DEC--TAN'
        self.wcs.wcs.crpix = 322, 151
        self.wcs.wcs.crval = 43, 23
        self.wcs.wcs.cdelt = -0.1, 0.1
        self.wcs.wcs.equinox = 2000.

        self.array = np.random.random((423, 344))

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

        ie = (0, 122, 233, 245, 344)
        je = (0, 44, 45, 333, 335, 423)

        views = []
        for i in range(4):
            for j in range(5):
                views.append((je[j], je[j+1], ie[i], ie[i+1]))

        return views

    @property
    def _overlapping_views(self):

        ie = (0, 122, 233, 245, 344)
        je = (0, 44, 45, 333, 335, 423)

        views = []
        for i in range(4):
            for j in range(5):
                views.append((je[j], je[j+1] + 10, ie[i], ie[i+1] + 10))

        return views

    @pytest.mark.parametrize('combine_function', ['mean', 'sum'])
    def test_coadd_no_overlap(self, combine_function):

        # Make sure that if all tiles are exactly non-overlapping, and
        # we use 'sum' or 'mean', we get the exact input array back.


        input_data = self._get_tiles(self._nonoverlapping_views)

        array, footprint = reproject_and_coadd(input_data, self.wcs, shape_out=self.array.shape,
                                               combine_function=combine_function, reproject_function=reproject_interp)

        assert_allclose(array, self.array, atol=1e-10)
        assert_equal(footprint, 1)

    def test_coadd_with_overlap(self):

        # Here we make the input tiles overlapping. We can only check the
        # mean, not the sum.

        input_data = self._get_tiles(self._overlapping_views)

        array, footprint = reproject_and_coadd(input_data, self.wcs, shape_out=self.array.shape,
                                               combine_function='mean', reproject_function=reproject_interp)

        assert_allclose(array, self.array, atol=1e-10)

    def test_coadd_background_matching(self):

        # Test out the background matching

        input_data = self._get_tiles(self._overlapping_views)

        for array, wcs in input_data:
            array += random.uniform(-3, 3)

        # First check that without background matching the arrays don't match

        array, footprint = reproject_and_coadd(input_data, self.wcs, shape_out=self.array.shape,
                                               combine_function='mean', reproject_function=reproject_interp)

        assert not np.allclose(array, self.array, atol=1e-10)

        # Now check that once the backgrounds are matched the values agree

        array, footprint = reproject_and_coadd(input_data, self.wcs, shape_out=self.array.shape,
                                               combine_function='mean', reproject_function=reproject_interp,
                                               match_background=True)

        # The absolute values of the two arrays will be offset since any solution
        # that reproduces the offsets between images is valid

        assert_allclose(array - np.mean(array),
                        self.array - np.mean(self.array), atol=1e-10)

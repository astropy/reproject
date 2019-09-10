# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from numpy.testing import assert_allclose
from astropy.io import fits
from astropy.wcs import WCS
from astropy.utils.data import get_pkg_data_filename
import pytest

from ..high_level import reproject_exact


class TestReprojectExact(object):

    def setup_class(self):

        self.header_in = fits.Header.fromtextfile(get_pkg_data_filename('../../tests/data/gc_ga.hdr'))
        self.header_out = fits.Header.fromtextfile(get_pkg_data_filename('../../tests/data/gc_eq.hdr'))

        self.header_out['NAXIS'] = 2
        self.header_out['NAXIS1'] = 600
        self.header_out['NAXIS2'] = 550

        self.array_in = np.ones((100, 100))

        self.wcs_in = WCS(self.header_in)
        self.wcs_out = WCS(self.header_out)

    def test_array_wcs(self):
        reproject_exact((self.array_in, self.wcs_in), self.wcs_out, shape_out=(200, 200))

    def test_array_header(self):
        reproject_exact((self.array_in, self.header_in), self.header_out)

    def test_parallel_option(self):

        reproject_exact((self.array_in, self.header_in), self.header_out, parallel=1)

        with pytest.raises(ValueError) as exc:
            reproject_exact((self.array_in, self.header_in), self.header_out, parallel=-1)
        assert exc.value.args[0] == "The number of processors to use must be strictly positive"


def test_identity():

    # Reproject an array and WCS to itself

    wcs = WCS(naxis=2)
    wcs.wcs.ctype = 'RA---TAN', 'DEC--TAN'
    wcs.wcs.crpix = 322, 151
    wcs.wcs.crval = 43, 23
    wcs.wcs.cdelt = -0.1, 0.1
    wcs.wcs.equinox = 2000.

    np.random.seed(1249)

    # First check a case where the values should agree to very good precision
    array_in = np.random.random((133, 223))
    array_out, footprint = reproject_exact((array_in, wcs), wcs,
                                           shape_out=array_in.shape)
    assert_allclose(array_out, array_in, atol=1e-10)

    # FIXME: As we make the array larger, the values still agree well but not
    # to as high precision, so ideally it would be good to track this down.
    array_in = np.random.random((423, 344))
    array_out, footprint = reproject_exact((array_in, wcs), wcs,
                                           shape_out=array_in.shape)
    assert_allclose(array_out, array_in, atol=1e-3)
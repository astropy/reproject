# Licensed under a 3-clause BSD style license - see LICENSE.rst

import warnings
import numpy as np
import pytest
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from astropy.wcs import WCS
from numpy.testing import assert_allclose

from ..high_level import reproject_exact


class TestReprojectExact:

    def setup_class(self):

        header_gal = get_pkg_data_filename('../../tests/data/gc_ga.hdr')
        header_equ = get_pkg_data_filename('../../tests/data/gc_eq.hdr')
        self.header_in = fits.Header.fromtextfile(header_gal)
        self.header_out = fits.Header.fromtextfile(header_equ)

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

    array_in = np.random.random((423, 344))
    array_out, footprint = reproject_exact((array_in, wcs), wcs,
                                           shape_out=array_in.shape)

    assert_allclose(array_out, array_in, atol=1e-10)


def test_reproject_precision_warning():

    for res in [0.1 / 3600, 0.01 / 3600]:

        wcs1 = WCS()
        wcs1.wcs.ctype = 'RA---TAN', 'DEC--TAN'
        wcs1.wcs.crval = 13, 80
        wcs1.wcs.crpix = 10., 10.
        wcs1.wcs.cdelt = res, res

        wcs2 = WCS()
        wcs2.wcs.ctype = 'RA---TAN', 'DEC--TAN'
        wcs2.wcs.crval = 13, 80
        wcs2.wcs.crpix = 3, 3
        wcs2.wcs.cdelt = 3 * res, 3 * res

        array = np.zeros((19, 19))
        array[9, 9] = 1

        if res < 0.05 / 3600:
            with pytest.warns(UserWarning, match='The reproject_exact function '
                                                 'currently has precision'):
                reproject_exact((array, wcs1), wcs2, shape_out=(5, 5))
        else:
            with warnings.catch_warnings(record=True) as w:
                reproject_exact((array, wcs1), wcs2, shape_out=(5, 5))
            assert len(w) == 0

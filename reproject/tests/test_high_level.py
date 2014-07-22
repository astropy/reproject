# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.utils.data import get_pkg_data_filename
from astropy.tests.helper import pytest

from .. import reproject

# TODO: add reference comparisons


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
            reproject(self.hdu_in, self.header_out)
        assert exc.value.args[0] == "Need to specify shape since output header does not contain complete shape information"
        
        reproject(self.hdu_in, self.header_out_size)

    def test_hdu_wcs(self):
        reproject(self.hdu_in, self.wcs_out, shape_out=self.shape_out)

    def test_array_wcs_header(self):
        
        with pytest.raises(ValueError) as exc:
            reproject((self.array_in, self.wcs_in), self.header_out)
        assert exc.value.args[0] == "Need to specify shape since output header does not contain complete shape information"
        
        reproject((self.array_in, self.wcs_in), self.header_out_size)
        
    def test_array_wcs_wcs(self):
        reproject((self.array_in, self.wcs_in), self.wcs_out, shape_out=self.shape_out)

    def test_array_header_header(self):
        reproject((self.array_in, self.header_in), self.header_out_size)

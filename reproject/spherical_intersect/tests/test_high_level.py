# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.utils.data import get_pkg_data_filename
from astropy.tests.helper import pytest

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

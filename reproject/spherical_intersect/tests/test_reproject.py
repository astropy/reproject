# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.utils.data import get_pkg_data_filename

from .. import reproject_2d

# TODO: add reference comparisons


def test_reproject_celestial_slices_2d():

    header_in = fits.Header.fromtextfile(get_pkg_data_filename('../../tests/data/gc_ga.hdr'))
    header_out = fits.Header.fromtextfile(get_pkg_data_filename('../../tests/data/gc_eq.hdr'))

    array_in = np.ones((100, 100))

    wcs_in = WCS(header_in)
    wcs_out = WCS(header_out)

    array_out = reproject_2d(array_in, wcs_in, wcs_out, (200, 200))

    fits.writeto('/tmp/test.fits', array_out, clobber=True)

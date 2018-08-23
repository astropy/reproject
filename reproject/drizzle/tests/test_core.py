# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os

import pytest

from drizzle import drizzle
from astropy.io import fits

from ..core import _reproject_drizzle
from ..high_level import reproject_drizzle
from reproject.interpolation.tests.test_core import array_footprint_to_hdulist

DATA = os.path.join(os.path.dirname(__file__), '..', '..', 'tests', 'data')

#@pytest.mark.array_compare()
#def test_drizzle():
#    """
#    Test drizzle
#    """
#    hdu_in = fits.open(os.path.join(DATA, 'drizzle_j8bt06nyq_flt.fits'))[0]
#    header_out = hdu_in.header.copy()
#
#    array_out, footprint_out = reproject_drizzle(hdu_in, header_out)
#    return array_footprint_to_hdulist(array_out, footprint_out, header_out)

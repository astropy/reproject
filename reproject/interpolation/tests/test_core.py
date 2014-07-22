# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from .. import reproject_2d, reproject_celestial_slices

DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))

# TODO: add reference comparisons


def test_reproject_celestial_slices_2d():

    header_in = fits.Header.fromtextfile(os.path.join(DATA, 'gc_ga.hdr'))
    header_out = fits.Header.fromtextfile(os.path.join(DATA, 'gc_eq.hdr'))

    array_in = np.ones((700, 690))

    wcs_in = WCS(header_in)
    wcs_out = WCS(header_out)

    array_out = reproject_celestial_slices(array_in, wcs_in, wcs_out, (660, 680))

    array_out_2d = reproject_2d(array_in, wcs_in, wcs_out, (660, 680))

    np.testing.assert_allclose(array_out, array_out_2d)


def test_reproject_celestial_slices_3d():

    header_in = fits.Header.fromtextfile(os.path.join(DATA, 'cube.hdr'))

    array_in = np.ones((200, 180))

    wcs_in = WCS(header_in)
    wcs_out = wcs_in.deepcopy()
    wcs_out.wcs.ctype = ['GLON-SIN', 'GLAT-SIN', wcs_in.wcs.ctype[2]]
    wcs_out.wcs.crval = [158.0501, -21.530282, wcs_in.wcs.crval[2]]
    wcs_out.wcs.crpix = [50., 50., wcs_in.wcs.crpix[2]]

    array_out = reproject_celestial_slices(array_in, wcs_in, wcs_out, (160, 170))

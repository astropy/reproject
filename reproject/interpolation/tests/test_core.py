# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.utils.data import get_pkg_data_filename

from ..core import _reproject, map_coordinates, get_input_pixels

# TODO: add reference comparisons


def test_reproject_slices_2d():

    header_in = fits.Header.fromtextfile(get_pkg_data_filename('../../tests/data/gc_ga.hdr'))
    header_out = fits.Header.fromtextfile(get_pkg_data_filename('../../tests/data/gc_eq.hdr'))

    array_in = np.ones((700, 690))

    wcs_in = WCS(header_in)
    wcs_out = WCS(header_out)

    _reproject(array_in, wcs_in, wcs_out, (660, 680))


def test_reproject_slices_3d():

    header_in = fits.Header.fromtextfile(get_pkg_data_filename('../../tests/data/cube.hdr'))

    array_in = np.ones((200, 180))

    wcs_in = WCS(header_in)
    wcs_out = wcs_in.deepcopy()
    wcs_out.wcs.ctype = ['GLON-SIN', 'GLAT-SIN', wcs_in.wcs.ctype[2]]
    wcs_out.wcs.crval = [158.0501, -21.530282, wcs_in.wcs.crval[2]]
    wcs_out.wcs.crpix = [50., 50., wcs_in.wcs.crpix[2]]

    _reproject(array_in, wcs_in, wcs_out, (160, 170))


def test_map_coordinates_rectangular():

    # Regression test for a bug that was due to the resetting of the output
    # of map_coordinates to be in the wrong x/y direction

    image = np.ones((3, 10))
    coords = np.array([(0, 1, 2), (1, 5, 9)])

    result = map_coordinates(image, coords)

    np.testing.assert_allclose(result, 1)

def test_get_input_pixels():
    pass
    header = fits.Header.fromtextfile(get_pkg_data_filename('../../tests/data/gc_eq.hdr'))
    result = get_input_pixels(WCS(header).celestial, WCS(header).celestial, [2,2])
    
    np.testing.assert_allclose(result,
                               np.array((np.array([[ 5.05906428e-12,   1.00000000e+00],
                                                   [-9.89075488e-12,   1.00000000e+00]]),
                                         np.array([[6.19593266e-12,  -4.26325641e-12],
                                                   [1.00000000e+00,   1.00000000e+00]])))
                              )

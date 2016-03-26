# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.utils.data import get_pkg_data_filename
from astropy.tests.helper import pytest

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
    header_in = fits.Header.fromtextfile(get_pkg_data_filename('../../tests/data/cube.hdr'))

    header_in['NAXIS1'] = 5
    header_in['NAXIS2'] = 4
    header_in['NAXIS3'] = 3

    header_out = header_in.copy()
    header_out['NAXIS3'] = 2
    header_out['CRPIX3'] -= 0.5
    
    w_in = WCS(header_in)
    w_out = WCS(header_out)
    x_out,y_out,z_out = get_input_pixels(w_in, w_out, [2,4,5])
    
    np.testing.assert_allclose(z_out,
                               np.array([np.ones([4,5])*0.5,
                                         np.ones([4,5])*1.5,])
                              )

def test_reproject_full_3d():

    header_in = fits.Header.fromtextfile(get_pkg_data_filename('../../tests/data/cube.hdr'))

    array_in = np.ones((3, 200, 180))

    wcs_in = WCS(header_in)
    wcs_out = wcs_in.deepcopy()
    wcs_out.wcs.ctype = ['GLON-SIN', 'GLAT-SIN', wcs_in.wcs.ctype[2]]
    wcs_out.wcs.crval = [158.0501, -21.530282, wcs_in.wcs.crval[2]]
    wcs_out.wcs.crpix = [50., 50., wcs_in.wcs.crpix[2]+0.5]

    _reproject(array_in, wcs_in, wcs_out, (3, 160, 170))

def test_reproject_3d_full_correctness():
    inp_cube = np.arange(3, dtype='float').repeat(4*5).reshape(3,4,5)

    header_in = fits.Header.fromtextfile(get_pkg_data_filename('../../tests/data/cube.hdr'))

    header_in['NAXIS1'] = 5
    header_in['NAXIS2'] = 4
    header_in['NAXIS3'] = 3

    header_out = header_in.copy()
    header_out['NAXIS3'] = 2
    header_out['CRPIX3'] -= 0.5
    
    wcs_in = WCS(header_in)
    wcs_out = WCS(header_out)

    out_cube, out_cube_valid = _reproject(inp_cube, wcs_in, wcs_out, (2, 4, 5))
    # we expect to be projecting from
    # inp_cube = np.arange(3, dtype='float').repeat(4*5).reshape(3,4,5)
    # to
    # inp_cube_interp = (inp_cube[:-1]+inp_cube[1:])/2.
    # which is confirmed by
    # map_coordinates(inp_cube.astype('float'), new_coords, order=1, cval=np.nan, mode='constant')
    # np.testing.assert_allclose(inp_cube_interp, map_coordinates(inp_cube.astype('float'), new_coords, order=1, cval=np.nan, mode='constant'))
    assert out_cube.shape == (2,4,5)

    np.testing.assert_allclose(out_cube[out_cube_valid.astype('bool')],
                               ((inp_cube[:-1]+inp_cube[1:])/2.)[out_cube_valid.astype('bool')])

def test_4d_fails():
    header_in = fits.Header.fromtextfile(get_pkg_data_filename('../../tests/data/cube.hdr'))

    header_in['NAXIS4'] = 4
    header_in['NAXIS'] = 4

    header_out = header_in.copy()
    w_in = WCS(header_in)
    w_out = WCS(header_out)

    with pytest.raises(ValueError) as ex:
        x_out,y_out,z_out = get_input_pixels(w_in, w_out, [2,4,5,6])
    assert str(ex.value) == ">3 dimensional cube"

def test_inequal_wcs_dims():
    inp_cube = np.arange(3, dtype='float').repeat(4*5).reshape(3,4,5)
    header_in = fits.Header.fromtextfile(get_pkg_data_filename('../../tests/data/cube.hdr'))

    header_out = header_in.copy()
    header_out['CTYPE3'] = 'VRAD'
    header_out['CUNIT3'] = 'm/s'
    header_in['CTYPE3'] = 'AWAV'
    header_in['CUNIT3'] = 'm'
    
    wcs_in = WCS(header_in)
    wcs_out = WCS(header_out)

    with pytest.raises(ValueError) as ex:
        out_cube, out_cube_valid = _reproject(inp_cube, wcs_in, wcs_out, (2, 4, 5))
    assert str(ex.value) == "The input and output WCS are not equivalent"

# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.utils.data import get_pkg_data_filename

from .. import reproject_celestial

# TODO: add reference comparisons

def assert_allclose_nan(array1, array2):

    np.testing.assert_allclose(np.isnan(array1),
                               np.isnan(array2))

    np.testing.assert_allclose(array1[~np.isnan(array1)],
                               array2[~np.isnan(array2)], rtol=1.e-6)

def test_reproject_celestial_slices_2d():

    header_in = fits.Header.fromtextfile(get_pkg_data_filename('../../tests/data/gc_ga.hdr'))
    header_out = fits.Header.fromtextfile(get_pkg_data_filename('../../tests/data/gc_eq.hdr'))

    array_in = np.ones((100, 100))

    wcs_in = WCS(header_in)
    wcs_out = WCS(header_out)

    array_out = reproject_celestial(array_in, wcs_in, wcs_out, (200, 200))

DATA = np.array([[1, 2],[3, 4]])

INPUT_HDR = """
WCSAXES =                    2 / Number of coordinate axes
CRPIX1  =              299.628 / Pixel coordinate of reference point
CRPIX2  =              299.394 / Pixel coordinate of reference point
CDELT1  =         -0.001666666 / [deg] Coordinate increment at reference point
CDELT2  =          0.001666666 / [deg] Coordinate increment at reference point
CUNIT1  = 'deg'                / Units of coordinate increment and value
CUNIT2  = 'deg'                / Units of coordinate increment and value
CTYPE1  = 'GLON-CAR'           / galactic longitude, plate caree projection
CTYPE2  = 'GLAT-CAR'           / galactic latitude, plate caree projection
CRVAL1  =                  0.0 / [deg] Coordinate value at reference point
CRVAL2  =                  0.0 / [deg] Coordinate value at reference point
LONPOLE =                  0.0 / [deg] Native longitude of celestial pole
LATPOLE =                 90.0 / [deg] Native latitude of celestial pole
"""

OUTPUT_HDR = """
WCSAXES =                    2 / Number of coordinate axes
CRPIX1  =                  2.5 / Pixel coordinate of reference point
CRPIX2  =                  2.5 / Pixel coordinate of reference point
CDELT1  =         -0.001500000 / [deg] Coordinate increment at reference point
CDELT2  =          0.001500000 / [deg] Coordinate increment at reference point
CUNIT1  = 'deg'                / Units of coordinate increment and value
CUNIT2  = 'deg'                / Units of coordinate increment and value
CTYPE1  = 'RA---TAN'           / Right ascension, gnomonic projection
CTYPE2  = 'DEC--TAN'           / Declination, gnomonic projection
CRVAL1  =        267.183880241 / [deg] Coordinate value at reference point
CRVAL2  =        -28.768527143 / [deg] Coordinate value at reference point
LONPOLE =                180.0 / [deg] Native longitude of celestial pole
LATPOLE =        -28.768527143 / [deg] Native latitude of celestial pole
EQUINOX =               2000.0 / [yr] Equinox of equatorial coordinates
"""

def test_reproject_celestial_accuracy():

    wcs_in = WCS(fits.Header.fromstring(INPUT_HDR, sep='\n'))
    wcs_out = WCS(fits.Header.fromstring(OUTPUT_HDR, sep='\n'))

    array1, footprint1 = reproject_celestial(DATA, wcs_in, wcs_out, (4, 4), _method='legacy')
    array2, footprint2 = reproject_celestial(DATA, wcs_in, wcs_out, (4, 4), _method='c', parallel=False)
    array3, footprint3 = reproject_celestial(DATA, wcs_in, wcs_out, (4, 4), _method='c', parallel=True)

    assert_allclose_nan(array1, array2)
    assert_allclose_nan(array1, array3)

    assert_allclose_nan(footprint1, footprint2)
    assert_allclose_nan(footprint1, footprint3)

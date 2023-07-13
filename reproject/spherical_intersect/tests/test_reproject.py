# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from astropy.wcs import WCS

from ...interpolation.tests.test_core import as_high_level_wcs
from ..core import _reproject_celestial


def test_reproject_celestial_slices_2d():
    header_in = fits.Header.fromtextfile(get_pkg_data_filename("../../tests/data/gc_ga.hdr"))
    header_out = fits.Header.fromtextfile(get_pkg_data_filename("../../tests/data/gc_eq.hdr"))

    array_in = np.ones((100, 100))

    wcs_in = WCS(header_in)
    wcs_out = WCS(header_out)

    _reproject_celestial(array_in, wcs_in, wcs_out, (200, 200))


DATA = np.array([[1, 2], [3, 4]], dtype=np.int64)

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

MONTAGE_REF = np.array(
    [
        [np.nan, 2.0, 2.0, np.nan],
        [1.0, 1.6768244, 3.35364754, 4.0],
        [1.0, 1.6461656, 3.32308315, 4.0],
        [np.nan, 3.0, 3.0, np.nan],
    ]
)


@pytest.mark.parametrize("wcsapi", (False, True))
def test_reproject_celestial_montage(wcsapi):
    # Accuracy compared to Montage

    wcs_in = WCS(fits.Header.fromstring(INPUT_HDR, sep="\n"))
    wcs_out = WCS(fits.Header.fromstring(OUTPUT_HDR, sep="\n"))

    if wcsapi:  # Enforce a pure wcsapi API
        wcs_in, wcs_out = as_high_level_wcs(wcs_in), as_high_level_wcs(wcs_out)

    array, footprint = _reproject_celestial(DATA, wcs_in, wcs_out, (4, 4))

    # TODO: improve agreement with Montage - at the moment agreement is ~10%
    np.testing.assert_allclose(array, MONTAGE_REF, rtol=0.09)


def test_reproject_flipping():
    # Regression test for a bug that caused issues when the WCS was oriented
    # in a way that meant polygon vertices were clockwise.

    wcs_in = WCS(fits.Header.fromstring(INPUT_HDR, sep="\n"))
    wcs_out = WCS(fits.Header.fromstring(OUTPUT_HDR, sep="\n"))
    array1, footprint1 = _reproject_celestial(DATA, wcs_in, wcs_out, (4, 4))

    # Repeat with an input that is flipped horizontally with the equivalent WCS
    wcs_in_flipped = WCS(fits.Header.fromstring(INPUT_HDR, sep="\n"))
    wcs_in_flipped.wcs.cdelt[0] = -wcs_in_flipped.wcs.cdelt[0]
    wcs_in_flipped.wcs.crpix[0] = 3 - wcs_in_flipped.wcs.crpix[0]
    array2, footprint2 = _reproject_celestial(DATA[:, ::-1], wcs_in_flipped, wcs_out, (4, 4))

    # Repeat with an output that is flipped horizontally with the equivalent WCS
    wcs_out_flipped = WCS(fits.Header.fromstring(OUTPUT_HDR, sep="\n"))
    wcs_out_flipped.wcs.cdelt[0] = -wcs_out_flipped.wcs.cdelt[0]
    wcs_out_flipped.wcs.crpix[0] = 5 - wcs_out_flipped.wcs.crpix[0]
    array3, footprint3 = _reproject_celestial(DATA, wcs_in, wcs_out_flipped, (4, 4))
    array3, footprint3 = array3[:, ::-1], footprint3[:, ::-1]

    np.testing.assert_allclose(array1, array2, rtol=1.0e-5)
    np.testing.assert_allclose(array1, array3, rtol=1.0e-5)

    np.testing.assert_allclose(footprint1, footprint2, rtol=3.0e-5)
    np.testing.assert_allclose(footprint1, footprint3, rtol=3.0e-5)

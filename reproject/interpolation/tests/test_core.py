# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import itertools

import numpy as np
from distutils.version import LooseVersion
from astropy.io import fits
from astropy.wcs import WCS
from astropy.utils.data import get_pkg_data_filename
from astropy.tests.helper import pytest

from ...array_utils import map_coordinates
from ..core_celestial import _reproject_celestial
from ..core_full import _reproject_full
from ..high_level import reproject_interp

# TODO: add reference comparisons


DATA = os.path.join(os.path.dirname(__file__), '..', '..', 'tests', 'data')


def array_footprint_to_hdulist(array, footprint, header):
    hdulist = fits.HDUList()
    hdulist.append(fits.PrimaryHDU(array, header))
    hdulist.append(fits.ImageHDU(footprint, header, name='footprint'))
    return hdulist


@pytest.mark.fits_compare()
def test_reproject_celestial_2d_gal2equ():
    """
    Test reprojection of a 2D celestial image, which includes a coordinate
    system conversion.
    """
    hdu_in = fits.open(os.path.join(DATA, 'galactic_2d.fits'))[0]
    header_out = hdu_in.header.copy()
    header_out['CTYPE1'] = 'RA---TAN'
    header_out['CTYPE2'] = 'DEC--TAN'
    header_out['CRVAL1'] = 266.39311
    header_out['CRVAL2'] = -28.939779
    array_out, footprint_out = reproject_interp(hdu_in, header_out)
    return array_footprint_to_hdulist(array_out, footprint_out, header_out)


# Note that we can't use independent_celestial_slices=True and reorder the
# axes, hence why we need to prepare the combinations in this way.
AXIS_ORDER = list(itertools.permutations((0, 1, 2)))
COMBINATIONS = [(True, (0, 1, 2))]
for axis_order in AXIS_ORDER:
    COMBINATIONS.append((False, axis_order))

@pytest.mark.fits_compare(single_reference=True)
@pytest.mark.parametrize(('indep_slices', 'axis_order'), tuple(COMBINATIONS))
def test_reproject_celestial_3d_equ2gal(indep_slices, axis_order):
    """
    Test reprojection of a 3D cube with celestial components, which includes a
    coordinate system conversion (the original header is in equatorial
    coordinates). We test using both the 'fast' method which assumes celestial
    slices are independent, and the 'full' method. We also scramble the input
    dimensions of the data and header to make sure that the reprojection can
    deal with this.
    """

    # Read in the input cube
    hdu_in = fits.open(os.path.join(DATA, 'equatorial_3d.fits'))[0]

    # Define the output header - this should be the same for all versions of
    # this test to make sure we can use a single reference file.
    header_out = hdu_in.header.copy()
    header_out['NAXIS1'] = 10
    header_out['NAXIS2'] = 9
    header_out['CTYPE1'] = 'GLON-SIN'
    header_out['CTYPE2'] = 'GLAT-SIN'
    header_out['CRVAL1'] = 163.16724
    header_out['CRVAL2'] = -15.777405
    header_out['CRPIX1'] = 6
    header_out['CRPIX2'] = 5

    # We now scramble the input axes
    if axis_order != (0, 1, 2):
        wcs_in = WCS(hdu_in.header)
        wcs_in = wcs_in.sub((3 - np.array(axis_order)[::-1]).tolist())
        hdu_in.header = wcs_in.to_header()
        hdu_in.data = np.transpose(hdu_in.data, axis_order)

    array_out, footprint_out = reproject_interp(hdu_in, header_out,
                                                independent_celestial_slices=indep_slices)
    return array_footprint_to_hdulist(array_out, footprint_out, header_out)


@pytest.mark.fits_compare()
def test_small_cutout():
    """
    Test reprojection of a cutout from a larger image (makes sure that the
    pre-reprojection cropping works)
    """
    hdu_in = fits.open(os.path.join(DATA, 'galactic_2d.fits'))[0]
    header_out = hdu_in.header.copy()
    header_out['NAXIS1'] = 10
    header_out['NAXIS2'] = 9
    header_out['CTYPE1'] = 'RA---TAN'
    header_out['CTYPE2'] = 'DEC--TAN'
    header_out['CRVAL1'] = 266.39311
    header_out['CRVAL2'] = -28.939779
    header_out['CRPIX1'] = 5.1
    header_out['CRPIX2'] = 4.7
    array_out, footprint_out = reproject_interp(hdu_in, header_out)
    return array_footprint_to_hdulist(array_out, footprint_out, header_out)


def test_small_cutout_outside():
    """
    Test reprojection of a cutout from a larger image - in this case the
    cutout is completely outside the region of the input image so we should
    take a shortcut that returns arrays of NaNs.
    """
    hdu_in = fits.open(os.path.join(DATA, 'galactic_2d.fits'))[0]
    header_out = hdu_in.header.copy()
    header_out['NAXIS1'] = 10
    header_out['NAXIS2'] = 9
    header_out['CTYPE1'] = 'RA---TAN'
    header_out['CTYPE2'] = 'DEC--TAN'
    header_out['CRVAL1'] = 216.39311
    header_out['CRVAL2'] = -21.939779
    header_out['CRPIX1'] = 5.1
    header_out['CRPIX2'] = 4.7
    array_out, footprint_out = reproject_interp(hdu_in, header_out)
    assert np.all(np.isnan(array_out))
    assert np.all(footprint_out == 0)


def test_celestial_mismatch_2d():
    """
    Make sure an error is raised if the input image has celestial WCS
    information and the output does not (and vice-versa). This example will
    use the _reproject_celestial route.
    """

    hdu_in = fits.open(os.path.join(DATA, 'galactic_2d.fits'))[0]

    header_out = hdu_in.header.copy()
    header_out['CTYPE1'] = 'APPLES'
    header_out['CTYPE2'] = 'ORANGES'

    data = hdu_in.data
    wcs1 = WCS(hdu_in.header)
    wcs2 = WCS(header_out)

    with pytest.raises(ValueError) as exc:
        array_out, footprint_out = reproject_interp((data, wcs1), wcs2, shape_out=(2, 2))
    assert exc.value.args[0] == "Input WCS has celestial components but output WCS does not"


def test_celestial_mismatch_3d():
    """
    Make sure an error is raised if the input image has celestial WCS
    information and the output does not (and vice-versa). This example will
    use the _reproject_full route.
    """

    hdu_in = fits.open(os.path.join(DATA, 'equatorial_3d.fits'))[0]

    header_out = hdu_in.header.copy()
    header_out['CTYPE1'] = 'APPLES'
    header_out['CTYPE2'] = 'ORANGES'
    header_out['CTYPE3'] = 'BANANAS'

    data = hdu_in.data
    wcs1 = WCS(hdu_in.header)
    wcs2 = WCS(header_out)

    with pytest.raises(ValueError) as exc:
        array_out, footprint_out = reproject_interp((data, wcs1), wcs2, shape_out=(1, 2, 3))
    assert exc.value.args[0] == "Input WCS has celestial components but output WCS does not"

    with pytest.raises(ValueError) as exc:
        array_out, footprint_out = reproject_interp((data, wcs2), wcs1, shape_out=(1, 2, 3))
    assert exc.value.args[0] == "Output WCS has celestial components but input WCS does not"


def test_spectral_mismatch_3d():
    """
    Make sure an error is raised if there are mismatches between the presence
    or type of spectral axis.
    """

    hdu_in = fits.open(os.path.join(DATA, 'equatorial_3d.fits'))[0]

    header_out = hdu_in.header.copy()
    header_out['CTYPE3'] = 'FREQ'
    header_out['CUNIT3'] = 'Hz'

    data = hdu_in.data
    wcs1 = WCS(hdu_in.header)
    wcs2 = WCS(header_out)

    with pytest.raises(ValueError) as exc:
        array_out, footprint_out = reproject_interp((data, wcs1), wcs2, shape_out=(1, 2, 3))
    assert exc.value.args[0] == "The input (VOPT) and output (FREQ) spectral coordinate types are not equivalent."

    header_out['CTYPE3'] = 'BANANAS'
    wcs2 = WCS(header_out)

    with pytest.raises(ValueError) as exc:
        array_out, footprint_out = reproject_interp((data, wcs1), wcs2, shape_out=(1, 2, 3))
    assert exc.value.args[0] == "Input WCS has a spectral component but output WCS does not"

    with pytest.raises(ValueError) as exc:
        array_out, footprint_out = reproject_interp((data, wcs2), wcs1, shape_out=(1, 2, 3))
    assert exc.value.args[0] == "Output WCS has a spectral component but input WCS does not"


def test_naxis_mismatch():
    """
    Make sure an error is raised if the input and output WCS have a different
    number of dimensions.
    """

    data = np.ones((3, 2, 2))
    wcs_in = WCS(naxis=3)
    wcs_out = WCS(naxis=2)

    with pytest.raises(ValueError) as exc:
        array_out, footprint_out = reproject_interp((data, wcs_in), wcs_out, shape_out=(1, 2))
    assert exc.value.args[0] == "Number of dimensions between input and output WCS should match"


def test_slice_reprojection():
    """
    Test case where only the slices change and the celestial projection doesn't
    """

    inp_cube = np.arange(3, dtype='float').repeat(4 * 5).reshape(3, 4, 5)

    header_in = fits.Header.fromtextfile(get_pkg_data_filename('../../tests/data/cube.hdr'))

    header_in['NAXIS1'] = 5
    header_in['NAXIS2'] = 4
    header_in['NAXIS3'] = 3

    header_out = header_in.copy()
    header_out['NAXIS3'] = 2
    header_out['CRPIX3'] -= 0.5

    wcs_in = WCS(header_in)
    wcs_out = WCS(header_out)

    out_cube, out_cube_valid = _reproject_full(inp_cube, wcs_in, wcs_out, (2, 4, 5))

    # we expect to be projecting from
    # inp_cube = np.arange(3, dtype='float').repeat(4*5).reshape(3,4,5)
    # to
    # inp_cube_interp = (inp_cube[:-1]+inp_cube[1:])/2.
    # which is confirmed by
    # map_coordinates(inp_cube.astype('float'), new_coords, order=1, cval=np.nan, mode='constant')
    # np.testing.assert_allclose(inp_cube_interp, map_coordinates(inp_cube.astype('float'), new_coords, order=1, cval=np.nan, mode='constant'))

    assert out_cube.shape == (2, 4, 5)
    assert out_cube_valid.sum() == 40.

    # We only check that the *valid* pixels are equal
    # but it's still nice to check that the "valid" array works as a mask
    np.testing.assert_allclose(out_cube[out_cube_valid.astype('bool')],
                               ((inp_cube[:-1] + inp_cube[1:]) / 2.)[out_cube_valid.astype('bool')])

    # Actually, I fixed it, so now we can test all
    np.testing.assert_allclose(out_cube, ((inp_cube[:-1] + inp_cube[1:]) / 2.))


def test_4d_fails():

    header_in = fits.Header.fromtextfile(get_pkg_data_filename('../../tests/data/cube.hdr'))

    header_in['NAXIS'] = 4

    header_out = header_in.copy()
    w_in = WCS(header_in)
    w_out = WCS(header_out)

    array_in = np.zeros((2, 3, 4, 5))

    with pytest.raises(ValueError) as ex:
        x_out, y_out, z_out = reproject_interp((array_in, w_in), w_out, shape_out=[2, 4, 5, 6])
    assert str(ex.value) == "Length of shape_out should match number of dimensions in wcs_out"


def test_inequal_wcs_dims():
    inp_cube = np.arange(3, dtype='float').repeat(4 * 5).reshape(3, 4, 5)
    header_in = fits.Header.fromtextfile(get_pkg_data_filename('../../tests/data/cube.hdr'))

    header_out = header_in.copy()
    header_out['CTYPE3'] = 'VRAD'
    header_out['CUNIT3'] = 'm/s'
    header_in['CTYPE3'] = 'STOKES'
    header_in['CUNIT3'] = ''

    wcs_out = WCS(header_out)

    with pytest.raises(ValueError) as ex:
        out_cube, out_cube_valid = reproject_interp((inp_cube, header_in), wcs_out, shape_out=(2, 4, 5))
    assert str(ex.value) == "Output WCS has a spectral component but input WCS does not"


def test_different_wcs_types():

    inp_cube = np.arange(3, dtype='float').repeat(4 * 5).reshape(3, 4, 5)
    header_in = fits.Header.fromtextfile(get_pkg_data_filename('../../tests/data/cube.hdr'))

    header_out = header_in.copy()
    header_out['CTYPE3'] = 'VRAD'
    header_out['CUNIT3'] = 'm/s'
    header_in['CTYPE3'] = 'VELO'
    header_in['CUNIT3'] = 'm/s'

    wcs_in = WCS(header_in)
    wcs_out = WCS(header_out)

    with pytest.raises(ValueError) as ex:
        out_cube, out_cube_valid = reproject_interp((inp_cube, header_in), wcs_out, shape_out=(2, 4, 5))
    assert str(ex.value) == ("The input (VELO) and output (VRAD) spectral "
                             "coordinate types are not equivalent.")

# TODO: add a test to check the units are the same.


def test_reproject_3d_celestial_correctness_ra2gal():

    inp_cube = np.arange(3, dtype='float').repeat(7 * 8).reshape(3, 7, 8)

    header_in = fits.Header.fromtextfile(get_pkg_data_filename('../../tests/data/cube.hdr'))

    header_in['NAXIS1'] = 8
    header_in['NAXIS2'] = 7
    header_in['NAXIS3'] = 3

    header_out = header_in.copy()
    header_out['CTYPE1'] = 'GLON-TAN'
    header_out['CTYPE2'] = 'GLAT-TAN'
    header_out['CRVAL1'] = 158.5644791
    header_out['CRVAL2'] = -21.59589875
    # make the cube a cutout approximately in the center of the other one, but smaller
    header_out['NAXIS1'] = 4
    header_out['CRPIX1'] = 2
    header_out['NAXIS2'] = 3
    header_out['CRPIX2'] = 1.5

    header_out['NAXIS3'] = 2
    header_out['CRPIX3'] -= 0.5

    wcs_in = WCS(header_in)
    wcs_out = WCS(header_out)

    out_cube, out_cube_valid = reproject_interp((inp_cube, wcs_in), wcs_out, shape_out=(2, 3, 4))

    assert out_cube.shape == (2, 3, 4)
    assert out_cube_valid.sum() == out_cube.size

    # only compare the spectral axis
    np.testing.assert_allclose(out_cube[:, 0, 0], ((inp_cube[:-1] + inp_cube[1:]) / 2.)[:, 0, 0])


def test_reproject_celestial_3d():
    """
    Test both full_reproject and slicewise reprojection. We use a case where the
    non-celestial slices are the same and therefore where both algorithms can
    work.
    """

    header_in = fits.Header.fromtextfile(get_pkg_data_filename('../../tests/data/cube.hdr'))

    array_in = np.ones((3, 200, 180))

    # TODO: here we can check that if we change the order of the dimensions in
    # the WCS, things still work properly

    wcs_in = WCS(header_in)
    wcs_out = wcs_in.deepcopy()
    wcs_out.wcs.ctype = ['GLON-SIN', 'GLAT-SIN', wcs_in.wcs.ctype[2]]
    wcs_out.wcs.crval = [158.0501, -21.530282, wcs_in.wcs.crval[2]]
    wcs_out.wcs.crpix = [50., 50., wcs_in.wcs.crpix[2] + 0.5]

    out_full, foot_full = _reproject_full(array_in, wcs_in, wcs_out, (3, 160, 170))

    out_celestial, foot_celestial = _reproject_celestial(array_in, wcs_in, wcs_out, (3, 160, 170))

    np.testing.assert_allclose(out_full, out_celestial)
    np.testing.assert_allclose(foot_full, foot_celestial)

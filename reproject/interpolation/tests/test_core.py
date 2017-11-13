# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import itertools

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.utils.data import get_pkg_data_filename
import pytest

from ..core_celestial import _reproject_celestial
from ..core_full import _reproject_full
from ..high_level import reproject_interp
from ..utils import rebin

# TODO: add reference comparisons


DATA = os.path.join(os.path.dirname(__file__), '..', '..', 'tests', 'data')


def array_footprint_to_hdulist(array, footprint, header):
    hdulist = fits.HDUList()
    hdulist.append(fits.PrimaryHDU(array, header))
    hdulist.append(fits.ImageHDU(footprint, header, name='footprint'))
    return hdulist


@pytest.mark.array_compare()
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


@pytest.mark.array_compare(single_reference=True)
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


@pytest.mark.array_compare()
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


def test_mwpan_car_to_mol():
    """
    Test reprojection of the Mellinger Milky Way Panorama from CAR to MOL,
    which was returning all NaNs due to a regression that was introduced in
    reproject 0.3 (https://github.com/astrofrog/reproject/pull/124).
    """
    hdu_in = fits.Header.fromtextfile(os.path.join(DATA, 'mwpan2_RGB_3600.hdr'))
    wcs_in = WCS(hdu_in, naxis=2)
    data_in = np.ones((hdu_in['NAXIS2'], hdu_in['NAXIS1']), dtype=np.float)
    header_out = fits.Header()
    header_out['NAXIS'] = 2
    header_out['NAXIS1'] = 360
    header_out['NAXIS2'] = 180
    header_out['CRPIX1'] = 180
    header_out['CRPIX2'] = 90
    header_out['CRVAL1'] = 0
    header_out['CRVAL2'] = 0
    header_out['CDELT1'] = -2 * np.sqrt(2) / np.pi
    header_out['CDELT2'] = 2 * np.sqrt(2) / np.pi
    header_out['CTYPE1'] = 'GLON-MOL'
    header_out['CTYPE2'] = 'GLAT-MOL'
    header_out['RADESYS'] = 'ICRS'
    array_out, footprint_out = reproject_interp((data_in, wcs_in), header_out)
    assert np.isfinite(array_out).any()


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

def test_reproject_rebin_nowcs():
    """
    Test both full_reproject and slicewise reprojection. We use a case where the
    non-celestial slices are the same and therefore where both algorithms can
    work.
    """

    # Get an example data set
    fits_file = fits.open(os.path.join(DATA, 'galactic_2d.fits'))
    data = fits_file[0].data

    # Do the rebinning at a sub-sample level of 2.
    data_rebinned = rebin(data, 2)

    data_expected = np.array([[7.19175387e-06, 8.35745686e-06, 8.77151797e-06, 8.13859879e-06,
         2.00283121e-05, 1.87805326e-05, 4.68220787e-05, 3.81425853e-05, 
         3.11206859e-05, 4.40583462e-05, 0.00000000e+00], 
         [  6.79642153e-06, 9.68905351e-06, 1.13317592e-05, 1.07324477e-05, 
         1.52752091e-05, 1.74659581e-05, 2.16226454e-05, 2.37359109e-05, 
         2.84797807e-05, 4.15648356e-05, 0.00000000e+00], 
         [  7.68768405e-06, 8.57485338e-06, 8.34446746e-06, 8.24502968e-06, 
         1.35071186e-05, 1.29909613e-05, 1.64359171e-05, 2.02391657e-05, 
         2.21153932e-05, 3.04823971e-05, 0.00000000e+00], 
         [  8.35000628e-06, 6.69123347e-06, 8.58000840e-06, 1.00389479e-05, 
         1.26938021e-05, 2.32635030e-05, 1.91833278e-05, 1.80794577e-05, 
         2.56445346e-05, 2.26606953e-05, 0.00000000e+00], 
         [  6.86031080e-06, 7.36491074e-06, 8.58933163e-06, 9.96438121e-06, 
         4.25930702e-05, 3.63348445e-05, 2.47167864e-05, 2.08366291e-05, 
         2.12481900e-05, 2.36051601e-05, 0.00000000e+00], 
         [  6.32698357e-06, 6.95157496e-06, 6.73984414e-06, 1.40516049e-05, 
         4.41167394e-05, 3.66262830e-05, 3.99278288e-05, 2.61872920e-05, 
         2.05276774e-05, 1.96867841e-05, 0.00000000e+00], 
         [  7.54499570e-06, 6.68639450e-06, 8.49540174e-06, 1.04629407e-05, 
         2.01836210e-05, 5.81512541e-05, 1.69402774e-04, 4.54312285e-05, 
         2.15097862e-05, 1.63309833e-05, 0.00000000e+00], 
         [  7.14042380e-06, 7.60676176e-06, 4.99871703e-06, 6.73289287e-06, 
         1.14446020e-05, 4.08667838e-05, 1.23506266e-04, 3.51971503e-05, 
         1.87233018e-05, 1.86361758e-05, 0.00000000e+00], 
         [  6.49218373e-06, 5.47602258e-06, 5.44384966e-06, 7.03356864e-06, 
         9.73041097e-06, 1.34041638e-05, 1.87683763e-05, 1.46828943e-05, 
         1.67347334e-05, 1.41952560e-05, 0.00000000e+00], 
         [  1.13113838e-05, 7.23780931e-06, 5.14646172e-06, 5.00229226e-06, 
         6.77710614e-06, 9.55357882e-06, 1.23373184e-05, 1.40258217e-05, 
         1.18025473e-05, 1.28170568e-05, 0.00000000e+00], 
         [  0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 
         0.00000000e+00, 0.00000000e+00, 0.00000000e+00]])

    # Check to make sure they are close
    np.testing.assert_allclose(data_rebinned[::2, ::2], data_expected)


def test_reproject_rebin_wcs():
    """
    Test both full_reproject and slicewise reprojection. We use a case where the
    non-celestial slices are the same and therefore where both algorithms can
    work.
    """

    from astropy.wcs import WCS

    # Get an example data set
    fits_file = fits.open(os.path.join(DATA, 'galactic_2d.fits'))
    data = fits_file[0].data
    wcs = WCS(fits_file[0])

    # Do the rebinning at a sub-sample level of 2.
    data_rebinned, wcs_rebinned = rebin(data, 2, wcs)

    data_expected = np.array([[5.02525836e-06, 8.64204139e-06, 8.32688875e-06, 8.56589941e-06,
          1.64098492e-05, 1.81917585e-05, 3.16653786e-05, 3.66636814e-05,
          2.83842310e-05, 3.59535370e-05, 5.57118801e-05],
        [  6.58651470e-06, 8.85325790e-06, 1.16944775e-05, 1.07865608e-05,
          1.55617618e-05, 1.63703553e-05, 2.27638575e-05, 2.31470913e-05,
          2.31784052e-05, 3.65043888e-05, 5.26272815e-05],
        [  8.94928507e-06, 8.00648832e-06, 8.95467656e-06, 8.32893511e-06,
          1.35752516e-05, 1.33324738e-05, 1.68891838e-05, 1.96405690e-05,
          1.87046371e-05, 2.69901884e-05, 3.70281487e-05],
        [  6.88575346e-06, 6.08080745e-06, 8.73690988e-06, 1.01955429e-05,
          1.22603024e-05, 2.20804868e-05, 1.78864138e-05, 1.81876130e-05,
          2.16477565e-05, 2.62393278e-05, 2.74925042e-05],
        [  6.49407184e-06, 8.51144250e-06, 9.29569160e-06, 1.00622847e-05,
          3.34361139e-05, 4.04107050e-05, 2.29261113e-05, 2.25640997e-05,
          2.29960697e-05, 2.15943655e-05, 2.50397679e-05],
        [  6.73715749e-06, 8.33323611e-06, 5.57969952e-06, 1.38713576e-05,
          4.82566538e-05, 5.11352482e-05, 3.72022150e-05, 2.73632668e-05,
          1.98471680e-05, 2.14072898e-05, 1.95206503e-05],
        [  5.55003999e-06, 7.01370300e-06, 6.58327008e-06, 1.33013846e-05,
          1.88961167e-05, 3.97709955e-05, 1.31088033e-04, 5.35792060e-05,
          2.59196804e-05, 2.27972596e-05, 2.13569801e-05],
        [  5.51730864e-06, 5.73142324e-06, 4.62910111e-06, 7.19284208e-06,
          9.90098306e-06, 4.72706233e-05, 2.23691561e-04, 6.21307263e-05,
          1.91485396e-05, 1.88092326e-05, 1.74927300e-05],
        [  6.03528997e-06, 4.50613970e-06, 5.49923607e-06, 6.51997880e-06,
          1.03725406e-05, 1.63900968e-05, 3.13550263e-05, 2.47909429e-05,
          1.70680378e-05, 1.99795122e-05, 1.63682926e-05],
        [  8.34748244e-06, 4.26154429e-06, 5.18316710e-06, 6.90082925e-06,
          9.46133150e-06, 1.13725982e-05, 1.40228904e-05, 1.64467456e-05,
          1.33941776e-05, 1.37421248e-05, 1.59778556e-05],
        [  2.55369778e-05, 1.04375995e-05, 8.73797399e-06, 7.66380526e-06,
          7.72606927e-06, 1.01555979e-05, 1.16204565e-05, 1.22227102e-05,
          1.30238668e-05, 1.34298270e-05, 1.70739640e-05]])

    # Check to make sure the data are close
    np.testing.assert_allclose(data_rebinned[::2, ::2], data_expected)

    # Check some primary elements of the rebinned WCS
    assert tuple(wcs_rebinned.wcs.ctype) == ('GLON-CAR', 'GLAT-CAR')
    np.testing.assert_allclose(wcs_rebinned.wcs.crval, (0., 0.))
    np.testing.assert_allclose(wcs_rebinned.wcs.cdelt, (-0.004,  0.004))
    np.testing.assert_allclose(wcs_rebinned.wcs.crpix, (10.,    10.75))

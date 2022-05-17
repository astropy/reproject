# Licensed under a 3-clause BSD style license - see LICENSE.rst

import itertools

import numpy as np
import pytest
from astropy import units as u
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from astropy.wcs import WCS
from astropy.wcs.wcs import FITSFixedWarning
from astropy.wcs.wcsapi import HighLevelWCSWrapper, SlicedLowLevelWCS
from numpy.testing import assert_allclose

from reproject.interpolation.high_level import reproject_interp
from reproject.tests.helpers import array_footprint_to_hdulist

# TODO: add reference comparisons


def as_high_level_wcs(wcs):
    return HighLevelWCSWrapper(SlicedLowLevelWCS(wcs, Ellipsis))


@pytest.mark.array_compare(single_reference=True)
@pytest.mark.parametrize('wcsapi', (False, True))
@pytest.mark.parametrize('roundtrip_coords', (False, True))
def test_reproject_celestial_2d_gal2equ(wcsapi, roundtrip_coords):
    """
    Test reprojection of a 2D celestial image, which includes a coordinate
    system conversion.
    """
    with fits.open(get_pkg_data_filename('data/galactic_2d.fits', package='reproject.tests')) as pf:
        hdu_in = pf[0]
        header_out = hdu_in.header.copy()
        header_out['CTYPE1'] = 'RA---TAN'
        header_out['CTYPE2'] = 'DEC--TAN'
        header_out['CRVAL1'] = 266.39311
        header_out['CRVAL2'] = -28.939779

        if wcsapi:  # Enforce a pure wcsapi API
            wcs_in, data_in = as_high_level_wcs(WCS(hdu_in.header)), hdu_in.data
            wcs_out = as_high_level_wcs(WCS(header_out))
            shape_out = header_out['NAXIS2'], header_out['NAXIS1']
            array_out, footprint_out = reproject_interp(
                    (data_in, wcs_in), wcs_out, shape_out=shape_out,
                    roundtrip_coords=roundtrip_coords)
        else:
            array_out, footprint_out = reproject_interp(
                    hdu_in, header_out, roundtrip_coords=roundtrip_coords)

    return array_footprint_to_hdulist(array_out, footprint_out, header_out)


# Note that we can't use independent_celestial_slices=True and reorder the
# axes, hence why we need to prepare the combinations in this way.
AXIS_ORDER = list(itertools.permutations((0, 1, 2)))
COMBINATIONS = []
for wcsapi in (False, True):
    for axis_order in AXIS_ORDER:
        COMBINATIONS.append((wcsapi, axis_order))


@pytest.mark.array_compare(single_reference=True)
@pytest.mark.parametrize(('wcsapi', 'axis_order'), tuple(COMBINATIONS))
@pytest.mark.parametrize('roundtrip_coords', (False, True))
def test_reproject_celestial_3d_equ2gal(wcsapi, axis_order, roundtrip_coords):
    """
    Test reprojection of a 3D cube with celestial components, which includes a
    coordinate system conversion (the original header is in equatorial
    coordinates). We test using both the 'fast' method which assumes celestial
    slices are independent, and the 'full' method. We also scramble the input
    dimensions of the data and header to make sure that the reprojection can
    deal with this.
    """

    # Read in the input cube
    with fits.open(
            get_pkg_data_filename('data/equatorial_3d.fits', package='reproject.tests')) as pf:
        hdu_in = pf[0]

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

        if wcsapi:  # Enforce a pure wcsapi API
            wcs_in, data_in = as_high_level_wcs(WCS(hdu_in.header)), hdu_in.data
            wcs_out = as_high_level_wcs(WCS(header_out))
            shape_out = header_out['NAXIS3'], header_out['NAXIS2'], header_out['NAXIS1']
            array_out, footprint_out = reproject_interp(
                    (data_in, wcs_in), wcs_out, shape_out=shape_out,
                    roundtrip_coords=roundtrip_coords)
        else:
            array_out, footprint_out = reproject_interp(
                    hdu_in, header_out, roundtrip_coords=roundtrip_coords)

    return array_footprint_to_hdulist(array_out, footprint_out, header_out)


@pytest.mark.array_compare(single_reference=True)
@pytest.mark.parametrize('wcsapi', (False, True))
@pytest.mark.parametrize('roundtrip_coords', (False, True))
def test_small_cutout(wcsapi, roundtrip_coords):
    """
    Test reprojection of a cutout from a larger image (makes sure that the
    pre-reprojection cropping works)
    """
    with fits.open(get_pkg_data_filename('data/galactic_2d.fits', package='reproject.tests')) as pf:
        hdu_in = pf[0]
        header_out = hdu_in.header.copy()
        header_out['NAXIS1'] = 10
        header_out['NAXIS2'] = 9
        header_out['CTYPE1'] = 'RA---TAN'
        header_out['CTYPE2'] = 'DEC--TAN'
        header_out['CRVAL1'] = 266.39311
        header_out['CRVAL2'] = -28.939779
        header_out['CRPIX1'] = 5.1
        header_out['CRPIX2'] = 4.7

        if wcsapi:  # Enforce a pure wcsapi API
            wcs_in, data_in = as_high_level_wcs(WCS(hdu_in.header)), hdu_in.data
            wcs_out = as_high_level_wcs(WCS(header_out))
            shape_out = header_out['NAXIS2'], header_out['NAXIS1']
            array_out, footprint_out = reproject_interp(
                    (data_in, wcs_in), wcs_out, shape_out=shape_out,
                    roundtrip_coords=roundtrip_coords)
        else:
            array_out, footprint_out = reproject_interp(
                    hdu_in, header_out, roundtrip_coords=roundtrip_coords)

    return array_footprint_to_hdulist(array_out, footprint_out, header_out)


@pytest.mark.parametrize('roundtrip_coords', (False, True))
def test_mwpan_car_to_mol(roundtrip_coords):
    """
    Test reprojection of the Mellinger Milky Way Panorama from CAR to MOL,
    which was returning all NaNs due to a regression that was introduced in
    reproject 0.3 (https://github.com/astrofrog/reproject/pull/124).
    """
    hdu_in = fits.Header.fromtextfile(
            get_pkg_data_filename('data/mwpan2_RGB_3600.hdr', package='reproject.tests'))
    with pytest.warns(FITSFixedWarning):
        wcs_in = WCS(hdu_in, naxis=2)
    data_in = np.ones((hdu_in['NAXIS2'], hdu_in['NAXIS1']), dtype=float)
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
    array_out, footprint_out = reproject_interp(
            (data_in, wcs_in), header_out, roundtrip_coords=roundtrip_coords)
    assert np.isfinite(array_out).any()


@pytest.mark.parametrize('roundtrip_coords', (False, True))
def test_small_cutout_outside(roundtrip_coords):
    """
    Test reprojection of a cutout from a larger image - in this case the
    cutout is completely outside the region of the input image so we should
    take a shortcut that returns arrays of NaNs.
    """
    with fits.open(get_pkg_data_filename('data/galactic_2d.fits', package='reproject.tests')) as pf:
        hdu_in = pf[0]
        header_out = hdu_in.header.copy()
        header_out['NAXIS1'] = 10
        header_out['NAXIS2'] = 9
        header_out['CTYPE1'] = 'RA---TAN'
        header_out['CTYPE2'] = 'DEC--TAN'
        header_out['CRVAL1'] = 216.39311
        header_out['CRVAL2'] = -21.939779
        header_out['CRPIX1'] = 5.1
        header_out['CRPIX2'] = 4.7
        array_out, footprint_out = reproject_interp(
                hdu_in, header_out, roundtrip_coords=roundtrip_coords)
    assert np.all(np.isnan(array_out))
    assert np.all(footprint_out == 0)


@pytest.mark.parametrize('roundtrip_coords', (False, True))
def test_celestial_mismatch_2d(roundtrip_coords):
    """
    Make sure an error is raised if the input image has celestial WCS
    information and the output does not (and vice-versa). This example will
    use the _reproject_celestial route.
    """
    with fits.open(get_pkg_data_filename('data/galactic_2d.fits', package='reproject.tests')) as pf:
        hdu_in = pf[0]

        header_out = hdu_in.header.copy()
        header_out['CTYPE1'] = 'APPLES'
        header_out['CTYPE2'] = 'ORANGES'

        data = hdu_in.data
        wcs1 = WCS(hdu_in.header)
        wcs2 = WCS(header_out)

        with pytest.raises(ValueError, match="Input WCS has celestial components but output WCS "
                           "does not"):
            array_out, footprint_out = reproject_interp(
                    (data, wcs1), wcs2, shape_out=(2, 2),
                    roundtrip_coords=roundtrip_coords)


@pytest.mark.parametrize('roundtrip_coords', (False, True))
def test_celestial_mismatch_3d(roundtrip_coords):
    """
    Make sure an error is raised if the input image has celestial WCS
    information and the output does not (and vice-versa). This example will
    use the _reproject_full route.
    """
    with fits.open(
            get_pkg_data_filename('data/equatorial_3d.fits', package='reproject.tests')) as pf:
        hdu_in = pf[0]

        header_out = hdu_in.header.copy()
        header_out['CTYPE1'] = 'APPLES'
        header_out['CTYPE2'] = 'ORANGES'
        header_out['CTYPE3'] = 'BANANAS'

        data = hdu_in.data
        wcs1 = WCS(hdu_in.header)
        wcs2 = WCS(header_out)

        with pytest.raises(ValueError, match="Input WCS has celestial components but output WCS "
                           "does not"):
            array_out, footprint_out = reproject_interp(
                    (data, wcs1), wcs2, shape_out=(1, 2, 3),
                    roundtrip_coords=roundtrip_coords)

        with pytest.raises(ValueError, match="Output WCS has celestial components but input WCS "
                           "does not"):
            array_out, footprint_out = reproject_interp(
                    (data, wcs2), wcs1, shape_out=(1, 2, 3),
                    roundtrip_coords=roundtrip_coords)


@pytest.mark.parametrize('roundtrip_coords', (False, True))
def test_spectral_mismatch_3d(roundtrip_coords):
    """
    Make sure an error is raised if there are mismatches between the presence
    or type of spectral axis.
    """
    with fits.open(
            get_pkg_data_filename('data/equatorial_3d.fits', package='reproject.tests')) as pf:
        hdu_in = pf[0]

        header_out = hdu_in.header.copy()
        header_out['CTYPE3'] = 'FREQ'
        header_out['CUNIT3'] = 'Hz'

        data = hdu_in.data
        wcs1 = WCS(hdu_in.header)
        wcs2 = WCS(header_out)

        with pytest.raises(ValueError, match=r"The input \(VOPT\) and output \(FREQ\) spectral "
                           r"coordinate types are not equivalent\."):
            array_out, footprint_out = reproject_interp(
                    (data, wcs1), wcs2, shape_out=(1, 2, 3),
                    roundtrip_coords=roundtrip_coords)

        header_out['CTYPE3'] = 'BANANAS'
        wcs2 = WCS(header_out)

        with pytest.raises(ValueError, match="Input WCS has a spectral component but output WCS "
                           "does not"):
            array_out, footprint_out = reproject_interp(
                    (data, wcs1), wcs2, shape_out=(1, 2, 3),
                    roundtrip_coords=roundtrip_coords)

        with pytest.raises(ValueError, match="Output WCS has a spectral component but input WCS "
                           "does not"):
            array_out, footprint_out = reproject_interp(
                    (data, wcs2), wcs1, shape_out=(1, 2, 3),
                    roundtrip_coords=roundtrip_coords)


@pytest.mark.parametrize('roundtrip_coords', (False, True))
def test_naxis_mismatch(roundtrip_coords):
    """
    Make sure an error is raised if the input and output WCS have a different
    number of dimensions.
    """
    data = np.ones((3, 2, 2))
    wcs_in = WCS(naxis=3)
    wcs_out = WCS(naxis=2)

    with pytest.raises(ValueError, match="Number of dimensions between input and output WCS "
                       "should match"):
        array_out, footprint_out = reproject_interp(
                (data, wcs_in), wcs_out, shape_out=(1, 2),
                roundtrip_coords=roundtrip_coords)


@pytest.mark.parametrize('roundtrip_coords', (False, True))
def test_slice_reprojection(roundtrip_coords):
    """
    Test case where only the slices change and the celestial projection doesn't
    """
    inp_cube = np.arange(3, dtype='float').repeat(4 * 5).reshape(3, 4, 5)

    header_in = fits.Header.fromtextfile(
        get_pkg_data_filename('data/cube.hdr', package='reproject.tests'))

    header_in['NAXIS1'] = 5
    header_in['NAXIS2'] = 4
    header_in['NAXIS3'] = 3

    header_out = header_in.copy()
    header_out['NAXIS3'] = 2
    header_out['CRPIX3'] -= 0.5

    wcs_in = WCS(header_in)
    wcs_out = WCS(header_out)

    out_cube, out_cube_valid = reproject_interp(
            (inp_cube, wcs_in), wcs_out, shape_out=(2, 4, 5),
            roundtrip_coords=roundtrip_coords)

    # we expect to be projecting from
    # inp_cube = np.arange(3, dtype='float').repeat(4*5).reshape(3,4,5)
    # to
    # inp_cube_interp = (inp_cube[:-1]+inp_cube[1:])/2.
    # which is confirmed by
    # map_coordinates(inp_cube.astype('float'), new_coords, order=1, cval=np.nan, mode='constant')
    # np.testing.assert_allclose(inp_cube_interp, map_coordinates(inp_cube.astype('float'),
    # new_coords, order=1, cval=np.nan, mode='constant'))

    assert out_cube.shape == (2, 4, 5)
    assert out_cube_valid.sum() == 40.

    # We only check that the *valid* pixels are equal
    # but it's still nice to check that the "valid" array works as a mask
    np.testing.assert_allclose(out_cube[out_cube_valid.astype('bool')],
                               ((inp_cube[:-1] + inp_cube[1:]) / 2.)[out_cube_valid.astype('bool')])

    # Actually, I fixed it, so now we can test all
    np.testing.assert_allclose(out_cube, ((inp_cube[:-1] + inp_cube[1:]) / 2.))


@pytest.mark.parametrize('roundtrip_coords', (False, True))
def test_4d_fails(roundtrip_coords):

    header_in = fits.Header.fromtextfile(
        get_pkg_data_filename('data/cube.hdr', package='reproject.tests'))

    header_in['NAXIS'] = 4

    header_out = header_in.copy()
    w_in = WCS(header_in)
    w_out = WCS(header_out)

    array_in = np.zeros((2, 3, 4, 5))

    with pytest.raises(ValueError, match="Length of shape_out should match number of dimensions "
                       "in wcs_out"):
        x_out, y_out, z_out = reproject_interp(
                (array_in, w_in), w_out, shape_out=[2, 4, 5, 6],
                roundtrip_coords=roundtrip_coords)


@pytest.mark.parametrize('roundtrip_coords', (False, True))
def test_inequal_wcs_dims(roundtrip_coords):
    inp_cube = np.arange(3, dtype='float').repeat(4 * 5).reshape(3, 4, 5)
    header_in = fits.Header.fromtextfile(
        get_pkg_data_filename('data/cube.hdr', package='reproject.tests'))

    header_out = header_in.copy()
    header_out['CTYPE3'] = 'VRAD'
    header_out['CUNIT3'] = 'm/s'
    header_in['CTYPE3'] = 'STOKES'
    header_in['CUNIT3'] = ''

    wcs_out = WCS(header_out)

    with pytest.raises(ValueError, match="Output WCS has a spectral component but input WCS "
                       "does not"):
        out_cube, out_cube_valid = reproject_interp(
                (inp_cube, header_in), wcs_out, shape_out=(2, 4, 5),
                roundtrip_coords=roundtrip_coords)


@pytest.mark.parametrize('roundtrip_coords', (False, True))
def test_different_wcs_types(roundtrip_coords):

    inp_cube = np.arange(3, dtype='float').repeat(4 * 5).reshape(3, 4, 5)
    header_in = fits.Header.fromtextfile(
        get_pkg_data_filename('data/cube.hdr', package='reproject.tests'))

    header_out = header_in.copy()
    header_out['CTYPE3'] = 'VRAD'
    header_out['CUNIT3'] = 'm/s'
    header_in['CTYPE3'] = 'VELO'
    header_in['CUNIT3'] = 'm/s'

    wcs_out = WCS(header_out)

    with pytest.raises(ValueError, match=r"The input \(VELO\) and output \(VRAD\) spectral "
                                         r"coordinate types are not equivalent\."):
        out_cube, out_cube_valid = reproject_interp(
                (inp_cube, header_in), wcs_out, shape_out=(2, 4, 5),
                roundtrip_coords=roundtrip_coords)

# TODO: add a test to check the units are the same.


@pytest.mark.parametrize('roundtrip_coords', (False, True))
def test_reproject_3d_celestial_correctness_ra2gal(roundtrip_coords):

    inp_cube = np.arange(3, dtype='float').repeat(7 * 8).reshape(3, 7, 8)

    header_in = fits.Header.fromtextfile(
        get_pkg_data_filename('data/cube.hdr', package='reproject.tests'))

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

    out_cube, out_cube_valid = reproject_interp(
            (inp_cube, wcs_in), wcs_out, shape_out=(2, 3, 4),
            roundtrip_coords=roundtrip_coords)

    assert out_cube.shape == (2, 3, 4)
    assert out_cube_valid.sum() == out_cube.size

    # only compare the spectral axis
    np.testing.assert_allclose(out_cube[:, 0, 0], ((inp_cube[:-1] + inp_cube[1:]) / 2.)[:, 0, 0])


@pytest.mark.parametrize('roundtrip_coords', (False, True))
def test_reproject_with_output_array(roundtrip_coords):
    """
    Test both full_reproject and slicewise reprojection. We use a case where the
    non-celestial slices are the same and therefore where both algorithms can
    work.
    """
    header_in = fits.Header.fromtextfile(
        get_pkg_data_filename('data/cube.hdr', package='reproject.tests'))

    array_in = np.ones((3, 200, 180))
    shape_out = (3, 160, 170)
    out_full = np.empty(shape_out)

    wcs_in = WCS(header_in)
    wcs_out = wcs_in.deepcopy()
    wcs_out.wcs.ctype = ['GLON-SIN', 'GLAT-SIN', wcs_in.wcs.ctype[2]]
    wcs_out.wcs.crval = [158.0501, -21.530282, wcs_in.wcs.crval[2]]
    wcs_out.wcs.crpix = [50., 50., wcs_in.wcs.crpix[2] + 0.4]

    # TODO when someone learns how to do it: make sure the memory isn't duplicated...
    returned_array = reproject_interp((array_in, wcs_in), wcs_out,
                                      output_array=out_full,
                                      return_footprint=False,
                                      roundtrip_coords=roundtrip_coords)

    assert out_full is returned_array


@pytest.mark.array_compare(single_reference=True)
@pytest.mark.parametrize('file_format', ['fits', 'asdf'])
def test_reproject_roundtrip(file_format):

    # Test the reprojection with solar data, which ensures that the masking of
    # pixels based on round-tripping works correctly. Using asdf is not just
    # about testing a different format but making sure that GWCS works.

    # The observer handling changed in 2.1.
    pytest.importorskip('sunpy', minversion='2.1.0')
    from sunpy.map import Map
    from sunpy.coordinates.ephemeris import get_body_heliographic_stonyhurst

    if file_format == 'fits':
        map_aia = Map(get_pkg_data_filename('data/aia_171_level1.fits', package='reproject.tests'))
        data = map_aia.data
        wcs = map_aia.wcs
        date = map_aia.date
        target_wcs = wcs.deepcopy()
    elif file_format == 'asdf':
        pytest.importorskip('astropy', minversion='4.0')
        pytest.importorskip('gwcs', minversion='0.12')
        asdf = pytest.importorskip('asdf')
        aia = asdf.open(
            get_pkg_data_filename('data/aia_171_level1.asdf', package='reproject.tests'))
        data = aia['data'][...]
        wcs = aia['wcs']
        date = wcs.output_frame.reference_frame.obstime
        target_wcs = Map(
            get_pkg_data_filename('data/aia_171_level1.fits',
                                  package='reproject.tests')).wcs.deepcopy()
    else:
        raise ValueError('file_format should be fits or asdf')

    # Reproject to an observer on Venus

    target_wcs.wcs.cdelt = ([24, 24]*u.arcsec).to(u.deg)
    target_wcs.wcs.crpix = [64, 64]
    venus = get_body_heliographic_stonyhurst('venus', date)
    target_wcs.wcs.aux.hgln_obs = venus.lon.to_value(u.deg)
    target_wcs.wcs.aux.hglt_obs = venus.lat.to_value(u.deg)
    target_wcs.wcs.aux.dsun_obs = venus.radius.to_value(u.m)

    output, footprint = reproject_interp((data, wcs), target_wcs, (128, 128))

    header_out = target_wcs.to_header()

    # ASTROPY_LT_40: astropy v4.0 introduced new default header keywords,
    # once we support only astropy 4.0 and later we can update the reference
    # data files and remove this section.
    for key in ('CRLN_OBS', 'CRLT_OBS', 'DSUN_OBS', 'HGLN_OBS', 'HGLT_OBS',
                'MJDREFF', 'MJDREFI', 'MJDREF', 'MJD-OBS', 'RSUN_REF'):
        header_out.pop(key, None)
    header_out['DATE-OBS'] = header_out['DATE-OBS'].replace('T', ' ')

    return array_footprint_to_hdulist(output, footprint, header_out)


@pytest.mark.parametrize('roundtrip_coords', (False, True))
def test_identity_with_offset(roundtrip_coords):

    # Reproject an array and WCS to itself but with a margin, which should
    # end up empty. This is a regression test for a bug that caused some
    # values to extend beyond the original footprint.

    wcs = WCS(naxis=2)
    wcs.wcs.ctype = 'RA---TAN', 'DEC--TAN'
    wcs.wcs.crpix = 322, 151
    wcs.wcs.crval = 43, 23
    wcs.wcs.cdelt = -0.1, 0.1
    wcs.wcs.equinox = 2000.

    array_in = np.random.random((233, 123))

    wcs_out = wcs.deepcopy()
    wcs_out.wcs.crpix += 1
    shape_out = (array_in.shape[0] + 2, array_in.shape[1] + 2)

    array_out, footprint = reproject_interp(
            (array_in, wcs), wcs_out, shape_out=shape_out,
            roundtrip_coords=roundtrip_coords)

    expected = np.pad(array_in, 1, 'constant', constant_values=np.nan)

    assert_allclose(expected, array_out, atol=1e-10)

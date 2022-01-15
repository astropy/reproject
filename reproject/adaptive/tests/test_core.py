# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
from itertools import product

import numpy as np
import pytest
from astropy import units as u
from astropy.wcs import WCS
from astropy.wcs.wcsapi import HighLevelWCSWrapper, SlicedLowLevelWCS
from numpy.testing import assert_allclose

from ..high_level import reproject_adaptive
from ...tests.helpers import array_footprint_to_hdulist

DATA = os.path.join(os.path.dirname(__file__), '..', '..', 'tests', 'data')


def as_high_level_wcs(wcs):
    return HighLevelWCSWrapper(SlicedLowLevelWCS(wcs, Ellipsis))


@pytest.mark.array_compare(single_reference=True)
@pytest.mark.parametrize('wcsapi', (False, True))
@pytest.mark.parametrize('center_jacobian', (False, True))
@pytest.mark.parametrize('roundtrip_coords', (False, True))
def test_reproject_adaptive_2d(wcsapi, center_jacobian, roundtrip_coords):

    # Set up initial array with pattern
    data_in = np.zeros((256, 256))
    data_in[::20, :] = 1
    data_in[:, ::20] = 1
    data_in[10::20, 10::20] = 1

    # Define a simple input WCS
    wcs_in = WCS(naxis=2)
    wcs_in.wcs.crpix = 128.5, 128.5
    wcs_in.wcs.cdelt = -0.01, 0.01

    # Define a lower resolution output WCS
    wcs_out = WCS(naxis=2)
    wcs_out.wcs.crpix = 30.5, 30.5
    wcs_out.wcs.cdelt = -0.0427, 0.0427

    header_out = wcs_out.to_header()

    if wcsapi:  # Enforce a pure wcsapi API
        wcs_in = as_high_level_wcs(wcs_in)
        wcs_out = as_high_level_wcs(wcs_out)

    array_out, footprint_out = reproject_adaptive(
            (data_in, wcs_in), wcs_out, shape_out=(60, 60),
            center_jacobian=center_jacobian,
            roundtrip_coords=roundtrip_coords)

    # Check that surface brightness is conserved in the unrotated case
    assert_allclose(np.nansum(data_in), np.nansum(array_out) * (256 / 60) ** 2, rtol=0.1)

    # ASTROPY_LT_40: astropy v4.0 introduced new default header keywords,
    # once we support only astropy 4.0 and later we can update the reference
    # data files and remove this section.
    for key in ('DATEREF', 'MJDREFF', 'MJDREFI', 'MJDREF', 'MJD-OBS'):
        header_out.pop(key, None)

    return array_footprint_to_hdulist(array_out, footprint_out, header_out)


@pytest.mark.array_compare(single_reference=True)
@pytest.mark.parametrize('center_jacobian', (False, True))
@pytest.mark.parametrize('roundtrip_coords', (False, True))
def test_reproject_adaptive_2d_rotated(center_jacobian, roundtrip_coords):

    # Set up initial array with pattern
    data_in = np.zeros((256, 256))
    data_in[::20, :] = 1
    data_in[:, ::20] = 1
    data_in[10::20, 10::20] = 1

    # Define a simple input WCS
    wcs_in = WCS(naxis=2)
    wcs_in.wcs.crpix = 128.5, 128.5
    wcs_in.wcs.cdelt = -0.01, 0.01

    # Define a lower resolution output WCS with rotation
    wcs_out = WCS(naxis=2)
    wcs_out.wcs.crpix = 30.5, 30.5
    wcs_out.wcs.cdelt = -0.0427, 0.0427
    wcs_out.wcs.pc = [[0.8, 0.2], [-0.2, 0.8]]

    header_out = wcs_out.to_header()

    array_out, footprint_out = reproject_adaptive(
            (data_in, wcs_in), wcs_out, shape_out=(60, 60),
            center_jacobian=center_jacobian,
            roundtrip_coords=roundtrip_coords)

    # ASTROPY_LT_40: astropy v4.0 introduced new default header keywords,
    # once we support only astropy 4.0 and later we can update the reference
    # data files and remove this section.
    for key in ('DATEREF', 'MJDREFF', 'MJDREFI', 'MJDREF', 'MJD-OBS'):
        header_out.pop(key, None)

    return array_footprint_to_hdulist(array_out, footprint_out, header_out)


@pytest.mark.parametrize('roundtrip_coords', (False, True))
@pytest.mark.parametrize('center_jacobian', (False, True))
@pytest.mark.parametrize('kernel', ('hann', 'gaussian'))
def test_reproject_adaptive_high_aliasing_potential_rotation(
        roundtrip_coords, center_jacobian, kernel):
    # Generate sample data with vertical stripes alternating with every column
    data_in = np.arange(40*40).reshape((40, 40))
    data_in = (data_in) % 2

    # Set up the input image coordinates, defining pixel coordinates as world
    # coordinates (with an offset)
    wcs_in = WCS(naxis=2)
    wcs_in.wcs.crpix = 21, 21
    wcs_in.wcs.crval = 0, 0
    wcs_in.wcs.cdelt = 1, 1

    # Set up the output image coordinates
    wcs_out = WCS(naxis=2)
    wcs_out.wcs.crpix = 3, 5
    wcs_out.wcs.crval = 0, 0
    wcs_out.wcs.cdelt = 2, 1

    array_out = reproject_adaptive((data_in, wcs_in),
                                   wcs_out, shape_out=(11, 6),
                                   return_footprint=False,
                                   roundtrip_coords=roundtrip_coords,
                                   center_jacobian=center_jacobian,
                                   kernel=kernel)

    # The CDELT1 value in wcs_out produces a down-sampling by a factor of two
    # along the output x axis. With the input image containing vertical lines
    # with values of zero or one, we should have uniform values of 0.5
    # throughout our output array.
    np.testing.assert_allclose(array_out, 0.5, rtol=0.001)

    # Within the transforms, the order of operations is:
    # input pixel coordinates -> input rotation -> input scaling
    # -> world coords -> output scaling -> output rotation
    # -> output pixel coordinates. So if we add a 90-degree rotation to the
    # output image, we're declaring that image-x is world-y and vice-versa, but
    # since we're rotating the already-downsampled image, no pixel values
    # should change
    angle = 90 * np.pi / 180
    wcs_out.wcs.pc = [[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]]
    array_out = reproject_adaptive((data_in, wcs_in),
                                   wcs_out, shape_out=(11, 6),
                                   return_footprint=False,
                                   roundtrip_coords=roundtrip_coords,
                                   center_jacobian=center_jacobian,
                                   kernel=kernel)
    np.testing.assert_allclose(array_out, 0.5, rtol=0.001)

    # But if we add a 90-degree rotation to the input coordinates, then when
    # our stretched output pixels are projected onto the input data, they will
    # be stretched along the stripes, rather than perpendicular to them, and so
    # we'll still see the alternating stripes in our output data---whether or
    # not wcs_out contains a rotation.
    # For these last two cases, we only use a Hann kernel, since the blurring
    # inherent with the Gaussian kernel makes the comparison more difficult.
    angle = 90 * np.pi / 180
    wcs_in.wcs.pc = [[np.cos(angle), -np.sin(angle)],
                     [np.sin(angle), np.cos(angle)]]
    array_out = reproject_adaptive((data_in, wcs_in),
                                   wcs_out, shape_out=(11, 6),
                                   return_footprint=False,
                                   roundtrip_coords=roundtrip_coords,
                                   center_jacobian=center_jacobian,
                                   kernel='hann')

    # Generate the expected pattern of alternating stripes
    data_ref = np.arange(array_out.shape[1]) % 2
    data_ref = np.vstack([data_ref] * array_out.shape[0])
    np.testing.assert_allclose(array_out, data_ref)

    # Clear the rotation in the output coordinates
    wcs_out.wcs.pc = [[1, 0], [0, 1]]
    array_out = reproject_adaptive((data_in, wcs_in),
                                   wcs_out, shape_out=(11, 6),
                                   return_footprint=False,
                                   roundtrip_coords=roundtrip_coords,
                                   center_jacobian=center_jacobian,
                                   kernel='hann')
    data_ref = np.arange(array_out.shape[0]) % 2
    data_ref = np.vstack([data_ref] * array_out.shape[1]).T
    np.testing.assert_allclose(array_out, data_ref)


@pytest.mark.parametrize('roundtrip_coords', (False, True))
@pytest.mark.parametrize('center_jacobian', (False, True))
def test_reproject_adaptive_high_aliasing_potential_shearing(
        roundtrip_coords, center_jacobian):
    # Generate sample data with vertical stripes alternating with every column
    data_in = np.arange(40*40).reshape((40, 40))
    data_in = (data_in) % 2

    # Set up the input image coordinates, defining pixel coordinates as world
    # coordinates (with an offset)
    wcs_in = WCS(naxis=2)
    wcs_in.wcs.crpix = 21, 21
    wcs_in.wcs.crval = 0, 0
    wcs_in.wcs.cdelt = 1, 1

    for shear_x in (-1, 0, 1):
        for shear_y in (-1, 0, 1):
            if shear_x == shear_y == 0:
                continue

            # Set up the output image coordinates, with shearing in both x
            # and y
            wcs_out = WCS(naxis=2)
            wcs_out.wcs.crpix = 3, 5
            wcs_out.wcs.crval = 0, 0
            wcs_out.wcs.cdelt = 1, 1
            wcs_out.wcs.pc = (
                    np.array([[1, shear_x], [0, 1]])
                    @ np.array([[1, 0], [shear_y, 1]]))

            # n.b. The Gaussian kernel is much better-behaved in this
            # particular scenario, so using it allows a much smaller tolerance
            # in the comparison. Likewise, the kernel width is boosted to 1.5
            # to achieve stronger anti-aliasing for this test.
            array_out = reproject_adaptive((data_in, wcs_in),
                                           wcs_out, shape_out=(11, 6),
                                           return_footprint=False,
                                           roundtrip_coords=False,
                                           center_jacobian=center_jacobian,
                                           kernel='gaussian',
                                           kernel_width=1.5)

            # We should get values close to 0.5 (an average of the 1s and 0s
            # in the input image). This is as opposed to values near 0 or 1,
            # which would indicate incorrect averaging of sampled points.
            np.testing.assert_allclose(array_out, 0.5, atol=0.02, rtol=0)


def test_reproject_adaptive_flux_conservation():
    # This is more than just testing the `conserve_flux` flag---the expectation
    # that flux should be conserved gives a very easy way to quantify how
    # correct an image is. We use that to run through a grid of affine
    # transformations, checking that the output is correct for each one.

    # Generate input data of a single point---this is the most unforgiving
    # setup, as there's no change for multiple pixels to average out if their
    # reprojected flux goes slightly above or below being conserved.
    data_in = np.zeros((20, 20))
    data_in[12, 13] = 1

    # Set up the input image coordinates, defining pixel coordinates as world
    # coordinates (with an offset)
    wcs_in = WCS(naxis=2)
    wcs_in.wcs.crpix = 11, 11
    wcs_in.wcs.crval = 0, 0
    wcs_in.wcs.cdelt = 1, 1

    wcs_out = WCS(naxis=2)
    wcs_out.wcs.crpix = 16, 16
    wcs_out.wcs.crval = 0, 0

    # Define our grid of affine transformations. Even with only a few values
    # for each parameter, this is by far the longest-running test in this file,
    # so it's tough to justify a finer grid.
    rotations = [0, 45, 80, 90]
    scales_x = np.logspace(-.2, .2, 3)
    scales_y = np.logspace(-.3, .3, 3)
    translations_x = np.linspace(0, 8, 3)
    translations_y = np.linspace(0, .42, 3)
    shears_x = np.linspace(-.7, .7, 3)
    shears_y = np.linspace(-.2, .2, 3)

    for rot, scale_x, scale_y, trans_x, trans_y, shear_x, shear_y in product(
            rotations, scales_x, scales_y, translations_x, translations_y,
            shears_x, shears_y):
        wcs_out.wcs.cdelt = scale_x, scale_y
        angle = rot * np.pi/180
        rot_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                               [np.sin(angle),  np.cos(angle)]])
        shear_x_matrix = np.array([[1, shear_x], [0, 1]])
        shear_y_matrix = np.array([[1, 0], [shear_y, 1]])
        wcs_out.wcs.pc = rot_matrix @ shear_x_matrix @ shear_y_matrix

        # The Gaussian kernel does a better job at flux conservation, so
        # choosing it here allows a tighter tolerance. Increasing the sample
        # region width also allows a tighter tolerance---less room for bugs to
        # hide!
        array_out = reproject_adaptive((data_in, wcs_in),
                                       wcs_out, shape_out=(30, 30),
                                       return_footprint=False,
                                       roundtrip_coords=False,
                                       center_jacobian=False,
                                       kernel='gaussian',
                                       sample_region_width=5,
                                       conserve_flux=True)

        # The degree of flux-conservation we end up seeing isn't necessarily
        # something that we can constrain a priori, so here we test for
        # conservation to within 0.4% because that's as good as it's getting
        # right now. This test is more about ensuring future bugs don't
        # accidentally worsen flux conservation than ensuring that some
        # required threshold is met.
        np.testing.assert_allclose(np.nansum(array_out), 1, rtol=0.004)


def prepare_test_data(file_format):
    pytest.importorskip('sunpy', minversion='2.1.0')
    from sunpy.map import Map
    from sunpy.coordinates.ephemeris import get_body_heliographic_stonyhurst

    if file_format == 'fits':
        map_aia = Map(os.path.join(DATA, 'aia_171_level1.fits'))
        data = map_aia.data
        wcs = map_aia.wcs
        date = map_aia.date
        target_wcs = wcs.deepcopy()
    elif file_format == 'asdf':
        pytest.importorskip('astropy', minversion='4.0')
        pytest.importorskip('gwcs', minversion='0.12')
        asdf = pytest.importorskip('asdf')
        aia = asdf.open(os.path.join(DATA, 'aia_171_level1.asdf'))
        data = aia['data'][...]
        wcs = aia['wcs']
        date = wcs.output_frame.reference_frame.obstime
        target_wcs = Map(os.path.join(DATA, 'aia_171_level1.fits')).wcs.deepcopy()
    else:
        raise ValueError('file_format should be fits or asdf')

    # Reproject to an observer on Venus

    target_wcs.wcs.cdelt = ([24, 24]*u.arcsec).to(u.deg)
    target_wcs.wcs.crpix = [64, 64]
    venus = get_body_heliographic_stonyhurst('venus', date)
    target_wcs.wcs.aux.hgln_obs = venus.lon.to_value(u.deg)
    target_wcs.wcs.aux.hglt_obs = venus.lat.to_value(u.deg)
    target_wcs.wcs.aux.dsun_obs = venus.radius.to_value(u.m)

    return data, wcs, target_wcs


@pytest.mark.array_compare(single_reference=True)
@pytest.mark.parametrize('file_format', ['fits', 'asdf'])
def test_reproject_adaptive_roundtrip(file_format):

    # Test the reprojection with solar data, which ensures that the masking of
    # pixels based on round-tripping works correctly. Using asdf is not just
    # about testing a different format but making sure that GWCS works.

    data, wcs, target_wcs = prepare_test_data(file_format)

    output, footprint = reproject_adaptive((data, wcs), target_wcs, (128, 128),
                                           center_jacobian=True)

    header_out = target_wcs.to_header()

    # ASTROPY_LT_40: astropy v4.0 introduced new default header keywords,
    # once we support only astropy 4.0 and later we can update the reference
    # data files and remove this section.
    for key in ('CRLN_OBS', 'CRLT_OBS', 'DSUN_OBS', 'HGLN_OBS', 'HGLT_OBS',
                'MJDREFF', 'MJDREFI', 'MJDREF', 'MJD-OBS', 'RSUN_REF'):
        header_out.pop(key, None)
    header_out['DATE-OBS'] = header_out['DATE-OBS'].replace('T', ' ')

    return array_footprint_to_hdulist(output, footprint, header_out)


@pytest.mark.array_compare()
def test_reproject_adaptive_uncentered_jacobian():

    # Explicitly test the uncentered-Jacobian path for a non-affine transform.
    # For this case, output pixels change by 6% at most, and usually much less.
    # (Though more nan pixels are present, as the uncentered calculation draws
    # in values from a bit further away.)

    data, wcs, target_wcs = prepare_test_data('fits')

    output, footprint = reproject_adaptive((data, wcs), target_wcs, (128, 128),
                                           center_jacobian=False)

    header_out = target_wcs.to_header()

    # ASTROPY_LT_40: astropy v4.0 introduced new default header keywords,
    # once we support only astropy 4.0 and later we can update the reference
    # data files and remove this section.
    for key in ('CRLN_OBS', 'CRLT_OBS', 'DSUN_OBS', 'HGLN_OBS', 'HGLT_OBS',
                'MJDREFF', 'MJDREFI', 'MJDREF', 'MJD-OBS', 'RSUN_REF'):
        header_out.pop(key, None)
    header_out['DATE-OBS'] = header_out['DATE-OBS'].replace('T', ' ')

    return array_footprint_to_hdulist(output, footprint, header_out)

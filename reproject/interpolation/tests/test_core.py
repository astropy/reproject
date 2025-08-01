# Licensed under a 3-clause BSD style license - see LICENSE.rst

import itertools

import dask.array as da
import numpy as np
import pytest
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from astropy.wcs import WCS
from astropy.wcs.wcs import FITSFixedWarning
from astropy.wcs.wcsapi import HighLevelWCSWrapper, SlicedLowLevelWCS
from numpy.testing import assert_allclose

from reproject.interpolation.high_level import reproject_interp
from reproject.tests.helpers import array_footprint_to_hdulist

# TODO: add reference comparisons


@pytest.fixture(
    params=[None, "memmap", "none"],
)
def dask_method(request):
    return request.param


def as_high_level_wcs(wcs):
    return HighLevelWCSWrapper(SlicedLowLevelWCS(wcs, Ellipsis))


@pytest.mark.array_compare(single_reference=True)
@pytest.mark.parametrize("wcsapi", (False, True))
@pytest.mark.parametrize("roundtrip_coords", (False, True))
@pytest.mark.remote_data
def test_reproject_celestial_2d_gal2equ(wcsapi, roundtrip_coords):
    """
    Test reprojection of a 2D celestial image, which includes a coordinate
    system conversion.
    """
    with fits.open(get_pkg_data_filename("data/galactic_2d.fits", package="reproject.tests")) as pf:
        hdu_in = pf[0]
        header_out = hdu_in.header.copy()
        header_out["CTYPE1"] = "RA---TAN"
        header_out["CTYPE2"] = "DEC--TAN"
        header_out["CRVAL1"] = 266.39311
        header_out["CRVAL2"] = -28.939779

        if wcsapi:  # Enforce a pure wcsapi API
            wcs_in, data_in = as_high_level_wcs(WCS(hdu_in.header)), hdu_in.data
            wcs_out = as_high_level_wcs(WCS(header_out))
            shape_out = header_out["NAXIS2"], header_out["NAXIS1"]
            array_out, footprint_out = reproject_interp(
                (data_in, wcs_in), wcs_out, shape_out=shape_out, roundtrip_coords=roundtrip_coords
            )
        else:
            array_out, footprint_out = reproject_interp(
                hdu_in, header_out, roundtrip_coords=roundtrip_coords
            )

    return array_footprint_to_hdulist(array_out, footprint_out, header_out)


# Note that we can't use independent_celestial_slices=True and reorder the
# axes, hence why we need to prepare the combinations in this way.
AXIS_ORDER = list(itertools.permutations((0, 1, 2)))
COMBINATIONS = []
for wcsapi in (False, True):
    for axis_order in AXIS_ORDER:
        COMBINATIONS.append((wcsapi, axis_order))


@pytest.mark.array_compare(single_reference=True)
@pytest.mark.parametrize(("wcsapi", "axis_order"), tuple(COMBINATIONS))
@pytest.mark.parametrize("roundtrip_coords", (False, True))
@pytest.mark.remote_data
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
        get_pkg_data_filename("data/equatorial_3d.fits", package="reproject.tests")
    ) as pf:
        hdu_in = pf[0]

        # Define the output header - this should be the same for all versions of
        # this test to make sure we can use a single reference file.
        header_out = hdu_in.header.copy()
        header_out["NAXIS1"] = 10
        header_out["NAXIS2"] = 9
        header_out["CTYPE1"] = "GLON-SIN"
        header_out["CTYPE2"] = "GLAT-SIN"
        header_out["CRVAL1"] = 163.16724
        header_out["CRVAL2"] = -15.777405
        header_out["CRPIX1"] = 6
        header_out["CRPIX2"] = 5

        # We now scramble the input axes
        if axis_order != (0, 1, 2):
            wcs_in = WCS(hdu_in.header)
            wcs_in = wcs_in.sub((3 - np.array(axis_order)[::-1]).tolist())
            hdu_in.header = wcs_in.to_header()
            hdu_in.data = np.transpose(hdu_in.data, axis_order)

        if wcsapi:  # Enforce a pure wcsapi API
            wcs_in, data_in = as_high_level_wcs(WCS(hdu_in.header)), hdu_in.data
            wcs_out = as_high_level_wcs(WCS(header_out))
            shape_out = header_out["NAXIS3"], header_out["NAXIS2"], header_out["NAXIS1"]
            array_out, footprint_out = reproject_interp(
                (data_in, wcs_in), wcs_out, shape_out=shape_out, roundtrip_coords=roundtrip_coords
            )
        else:
            array_out, footprint_out = reproject_interp(
                hdu_in, header_out, roundtrip_coords=roundtrip_coords
            )

    return array_footprint_to_hdulist(array_out, footprint_out, header_out)


@pytest.mark.array_compare(single_reference=True)
@pytest.mark.parametrize("wcsapi", (False, True))
@pytest.mark.parametrize("roundtrip_coords", (False, True))
@pytest.mark.remote_data
def test_small_cutout(wcsapi, roundtrip_coords):
    """
    Test reprojection of a cutout from a larger image (makes sure that the
    pre-reprojection cropping works)
    """
    with fits.open(get_pkg_data_filename("data/galactic_2d.fits", package="reproject.tests")) as pf:
        hdu_in = pf[0]
        header_out = hdu_in.header.copy()
        header_out["NAXIS1"] = 10
        header_out["NAXIS2"] = 9
        header_out["CTYPE1"] = "RA---TAN"
        header_out["CTYPE2"] = "DEC--TAN"
        header_out["CRVAL1"] = 266.39311
        header_out["CRVAL2"] = -28.939779
        header_out["CRPIX1"] = 5.1
        header_out["CRPIX2"] = 4.7

        if wcsapi:  # Enforce a pure wcsapi API
            wcs_in, data_in = as_high_level_wcs(WCS(hdu_in.header)), hdu_in.data
            wcs_out = as_high_level_wcs(WCS(header_out))
            shape_out = header_out["NAXIS2"], header_out["NAXIS1"]
            array_out, footprint_out = reproject_interp(
                (data_in, wcs_in), wcs_out, shape_out=shape_out, roundtrip_coords=roundtrip_coords
            )
        else:
            array_out, footprint_out = reproject_interp(
                hdu_in, header_out, roundtrip_coords=roundtrip_coords
            )

    return array_footprint_to_hdulist(array_out, footprint_out, header_out)


@pytest.mark.parametrize("roundtrip_coords", (False, True))
@pytest.mark.remote_data
def test_mwpan_car_to_mol(roundtrip_coords):
    """
    Test reprojection of the Mellinger Milky Way Panorama from CAR to MOL,
    which was returning all NaNs due to a regression that was introduced in
    reproject 0.3 (https://github.com/astrofrog/reproject/pull/124).
    """
    hdu_in = fits.Header.fromtextfile(
        get_pkg_data_filename("data/mwpan2_RGB_3600.hdr", package="reproject.tests")
    )
    with pytest.warns(FITSFixedWarning):
        wcs_in = WCS(hdu_in, naxis=2)
    data_in = np.ones((hdu_in["NAXIS2"], hdu_in["NAXIS1"]), dtype=float)
    header_out = fits.Header()
    header_out["NAXIS"] = 2
    header_out["NAXIS1"] = 360
    header_out["NAXIS2"] = 180
    header_out["CRPIX1"] = 180
    header_out["CRPIX2"] = 90
    header_out["CRVAL1"] = 0
    header_out["CRVAL2"] = 0
    header_out["CDELT1"] = -2 * np.sqrt(2) / np.pi
    header_out["CDELT2"] = 2 * np.sqrt(2) / np.pi
    header_out["CTYPE1"] = "GLON-MOL"
    header_out["CTYPE2"] = "GLAT-MOL"
    header_out["RADESYS"] = "ICRS"
    array_out, footprint_out = reproject_interp(
        (data_in, wcs_in), header_out, roundtrip_coords=roundtrip_coords
    )
    assert np.isfinite(array_out).any()


@pytest.mark.parametrize("roundtrip_coords", (False, True))
@pytest.mark.remote_data
def test_small_cutout_outside(roundtrip_coords):
    """
    Test reprojection of a cutout from a larger image - in this case the
    cutout is completely outside the region of the input image so we should
    take a shortcut that returns arrays of NaNs.
    """
    with fits.open(get_pkg_data_filename("data/galactic_2d.fits", package="reproject.tests")) as pf:
        hdu_in = pf[0]
        header_out = hdu_in.header.copy()
        header_out["NAXIS1"] = 10
        header_out["NAXIS2"] = 9
        header_out["CTYPE1"] = "RA---TAN"
        header_out["CTYPE2"] = "DEC--TAN"
        header_out["CRVAL1"] = 216.39311
        header_out["CRVAL2"] = -21.939779
        header_out["CRPIX1"] = 5.1
        header_out["CRPIX2"] = 4.7
        array_out, footprint_out = reproject_interp(
            hdu_in, header_out, roundtrip_coords=roundtrip_coords
        )
    assert np.all(np.isnan(array_out))
    assert np.all(footprint_out == 0)


@pytest.mark.parametrize("roundtrip_coords", (False, True))
@pytest.mark.remote_data
def test_celestial_mismatch_2d(roundtrip_coords):
    """
    Make sure an error is raised if the input image has celestial WCS
    information and the output does not (and vice-versa). This example will
    use the _reproject_celestial route.
    """
    with fits.open(get_pkg_data_filename("data/galactic_2d.fits", package="reproject.tests")) as pf:
        hdu_in = pf[0]

        header_out = hdu_in.header.copy()
        header_out["CTYPE1"] = "APPLES"
        header_out["CTYPE2"] = "ORANGES"

        data = hdu_in.data
        wcs1 = WCS(hdu_in.header)
        wcs2 = WCS(header_out)

        with pytest.raises(
            ValueError, match="Input WCS has celestial components but output WCS does not"
        ):
            array_out, footprint_out = reproject_interp(
                (data, wcs1), wcs2, shape_out=(2, 2), roundtrip_coords=roundtrip_coords
            )


@pytest.mark.parametrize("roundtrip_coords", (False, True))
@pytest.mark.remote_data
def test_celestial_mismatch_3d(roundtrip_coords):
    """
    Make sure an error is raised if the input image has celestial WCS
    information and the output does not (and vice-versa). This example will
    use the _reproject_full route.
    """
    with fits.open(
        get_pkg_data_filename("data/equatorial_3d.fits", package="reproject.tests")
    ) as pf:
        hdu_in = pf[0]

        header_out = hdu_in.header.copy()
        header_out["CTYPE1"] = "APPLES"
        header_out["CTYPE2"] = "ORANGES"
        header_out["CTYPE3"] = "BANANAS"

        data = hdu_in.data
        wcs1 = WCS(hdu_in.header)
        wcs2 = WCS(header_out)

        with pytest.raises(
            ValueError, match="Input WCS has celestial components but output WCS does not"
        ):
            array_out, footprint_out = reproject_interp(
                (data, wcs1), wcs2, shape_out=(1, 2, 3), roundtrip_coords=roundtrip_coords
            )

        with pytest.raises(
            ValueError, match="Output WCS has celestial components but input WCS does not"
        ):
            array_out, footprint_out = reproject_interp(
                (data, wcs2), wcs1, shape_out=(1, 2, 3), roundtrip_coords=roundtrip_coords
            )


@pytest.mark.parametrize("roundtrip_coords", (False, True))
@pytest.mark.remote_data
def test_spectral_mismatch_3d(roundtrip_coords):
    """
    Make sure an error is raised if there are mismatches between the presence
    or type of spectral axis.
    """
    with fits.open(
        get_pkg_data_filename("data/equatorial_3d.fits", package="reproject.tests")
    ) as pf:
        hdu_in = pf[0]

        header_out = hdu_in.header.copy()
        header_out["CTYPE3"] = "FREQ"
        header_out["CUNIT3"] = "Hz"

        data = hdu_in.data
        wcs1 = WCS(hdu_in.header)
        wcs2 = WCS(header_out)

        with pytest.raises(
            ValueError,
            match=r"The input \(VOPT\) and output \(FREQ\) spectral "
            r"coordinate types are not equivalent\.",
        ):
            array_out, footprint_out = reproject_interp(
                (data, wcs1), wcs2, shape_out=(1, 2, 3), roundtrip_coords=roundtrip_coords
            )

        header_out["CTYPE3"] = "BANANAS"
        wcs2 = WCS(header_out)

        with pytest.raises(
            ValueError, match="Input WCS has a spectral component but output WCS does not"
        ):
            array_out, footprint_out = reproject_interp(
                (data, wcs1), wcs2, shape_out=(1, 2, 3), roundtrip_coords=roundtrip_coords
            )

        with pytest.raises(
            ValueError, match="Output WCS has a spectral component but input WCS does not"
        ):
            array_out, footprint_out = reproject_interp(
                (data, wcs2), wcs1, shape_out=(1, 2, 3), roundtrip_coords=roundtrip_coords
            )


@pytest.mark.parametrize("roundtrip_coords", (False, True))
def test_naxis_mismatch(roundtrip_coords):
    """
    Make sure an error is raised if the input and output WCS have a different
    number of dimensions.
    """
    data = np.ones((3, 2, 2))
    wcs_in = WCS(naxis=3)
    wcs_out = WCS(naxis=2)

    with pytest.raises(
        ValueError, match="Number of dimensions in input and output WCS should match"
    ):
        array_out, footprint_out = reproject_interp(
            (data, wcs_in), wcs_out, shape_out=(1, 2), roundtrip_coords=roundtrip_coords
        )


@pytest.mark.parametrize("roundtrip_coords", (False, True))
@pytest.mark.remote_data
def test_slice_reprojection(roundtrip_coords):
    """
    Test case where only the slices change and the celestial projection doesn't
    """
    inp_cube = np.arange(3, dtype="float").repeat(4 * 5).reshape(3, 4, 5)

    header_in = fits.Header.fromtextfile(
        get_pkg_data_filename("data/cube.hdr", package="reproject.tests")
    )

    header_in["NAXIS1"] = 5
    header_in["NAXIS2"] = 4
    header_in["NAXIS3"] = 3

    header_out = header_in.copy()
    header_out["NAXIS3"] = 2
    header_out["CRPIX3"] -= 0.5

    wcs_in = WCS(header_in)
    wcs_out = WCS(header_out)

    out_cube, out_cube_valid = reproject_interp(
        (inp_cube, wcs_in), wcs_out, shape_out=(2, 4, 5), roundtrip_coords=roundtrip_coords
    )

    # we expect to be projecting from
    # inp_cube = np.arange(3, dtype='float').repeat(4*5).reshape(3,4,5)
    # to
    # inp_cube_interp = (inp_cube[:-1]+inp_cube[1:])/2.
    # which is confirmed by
    # map_coordinates(inp_cube.astype('float'), new_coords, order=1, cval=np.nan, mode='constant')
    # np.testing.assert_allclose(inp_cube_interp, map_coordinates(inp_cube.astype('float'),
    # new_coords, order=1, cval=np.nan, mode='constant'))

    assert out_cube.shape == (2, 4, 5)
    assert out_cube_valid.sum() == 40.0

    # We only check that the *valid* pixels are equal
    # but it's still nice to check that the "valid" array works as a mask
    np.testing.assert_allclose(
        out_cube[out_cube_valid.astype("bool")],
        ((inp_cube[:-1] + inp_cube[1:]) / 2.0)[out_cube_valid.astype("bool")],
    )

    # Actually, I fixed it, so now we can test all
    np.testing.assert_allclose(out_cube, ((inp_cube[:-1] + inp_cube[1:]) / 2.0))


@pytest.mark.parametrize("roundtrip_coords", (False, True))
@pytest.mark.remote_data
def test_inequal_wcs_dims(roundtrip_coords):
    inp_cube = np.arange(3, dtype="float").repeat(4 * 5).reshape(3, 4, 5)
    header_in = fits.Header.fromtextfile(
        get_pkg_data_filename("data/cube.hdr", package="reproject.tests")
    )

    header_out = header_in.copy()
    header_out["CTYPE3"] = "VRAD"
    header_out["CUNIT3"] = "m/s"
    header_in["CTYPE3"] = "STOKES"
    header_in["CUNIT3"] = ""

    wcs_out = WCS(header_out)

    with pytest.raises(
        ValueError, match="Output WCS has a spectral component but input WCS does not"
    ):
        out_cube, out_cube_valid = reproject_interp(
            (inp_cube, header_in), wcs_out, shape_out=(2, 4, 5), roundtrip_coords=roundtrip_coords
        )


@pytest.mark.parametrize("roundtrip_coords", (False, True))
@pytest.mark.remote_data
def test_different_wcs_types(roundtrip_coords):
    inp_cube = np.arange(3, dtype="float").repeat(4 * 5).reshape(3, 4, 5)
    header_in = fits.Header.fromtextfile(
        get_pkg_data_filename("data/cube.hdr", package="reproject.tests")
    )

    header_out = header_in.copy()
    header_out["CTYPE3"] = "VRAD"
    header_out["CUNIT3"] = "m/s"
    header_in["CTYPE3"] = "VELO"
    header_in["CUNIT3"] = "m/s"

    wcs_out = WCS(header_out)

    with pytest.raises(
        ValueError,
        match=r"The input \(VELO\) and output \(VRAD\) spectral "
        r"coordinate types are not equivalent\.",
    ):
        out_cube, out_cube_valid = reproject_interp(
            (inp_cube, header_in), wcs_out, shape_out=(2, 4, 5), roundtrip_coords=roundtrip_coords
        )


# TODO: add a test to check the units are the same.


@pytest.mark.parametrize("roundtrip_coords", (False, True))
@pytest.mark.remote_data
def test_reproject_3d_celestial_correctness_ra2gal(roundtrip_coords):
    inp_cube = np.arange(3, dtype="float").repeat(7 * 8).reshape(3, 7, 8)

    header_in = fits.Header.fromtextfile(
        get_pkg_data_filename("data/cube.hdr", package="reproject.tests")
    )

    header_in["NAXIS1"] = 8
    header_in["NAXIS2"] = 7
    header_in["NAXIS3"] = 3

    header_out = header_in.copy()
    header_out["CTYPE1"] = "GLON-TAN"
    header_out["CTYPE2"] = "GLAT-TAN"
    header_out["CRVAL1"] = 158.5644791
    header_out["CRVAL2"] = -21.59589875
    # make the cube a cutout approximately in the center of the other one, but smaller
    header_out["NAXIS1"] = 4
    header_out["CRPIX1"] = 2
    header_out["NAXIS2"] = 3
    header_out["CRPIX2"] = 1.5

    header_out["NAXIS3"] = 2
    header_out["CRPIX3"] -= 0.5

    wcs_in = WCS(header_in)
    wcs_out = WCS(header_out)

    out_cube, out_cube_valid = reproject_interp(
        (inp_cube, wcs_in), wcs_out, shape_out=(2, 3, 4), roundtrip_coords=roundtrip_coords
    )

    assert out_cube.shape == (2, 3, 4)
    assert out_cube_valid.sum() == out_cube.size

    # only compare the spectral axis
    np.testing.assert_allclose(out_cube[:, 0, 0], ((inp_cube[:-1] + inp_cube[1:]) / 2.0)[:, 0, 0])


@pytest.mark.parametrize("roundtrip_coords", (False, True))
@pytest.mark.remote_data
def test_reproject_with_output_array(roundtrip_coords):
    """
    Test both full_reproject and slicewise reprojection. We use a case where the
    non-celestial slices are the same and therefore where both algorithms can
    work.
    """
    header_in = fits.Header.fromtextfile(
        get_pkg_data_filename("data/cube.hdr", package="reproject.tests")
    )

    array_in = np.ones((3, 200, 180))
    shape_out = (3, 160, 170)
    out_full = np.empty(shape_out)

    wcs_in = WCS(header_in)
    wcs_out = wcs_in.deepcopy()
    wcs_out.wcs.ctype = ["GLON-SIN", "GLAT-SIN", wcs_in.wcs.ctype[2]]
    wcs_out.wcs.crval = [158.0501, -21.530282, wcs_in.wcs.crval[2]]
    wcs_out.wcs.crpix = [50.0, 50.0, wcs_in.wcs.crpix[2] + 0.4]

    # TODO when someone learns how to do it: make sure the memory isn't duplicated...
    returned_array = reproject_interp(
        (array_in, wcs_in),
        wcs_out,
        output_array=out_full,
        return_footprint=False,
        roundtrip_coords=roundtrip_coords,
    )

    assert out_full is returned_array


@pytest.mark.array_compare(single_reference=True)
@pytest.mark.remote_data
def test_reproject_roundtrip(aia_test_data):
    # Test the reprojection with solar data, which ensures that the masking of
    # pixels based on round-tripping works correctly. Using asdf is not just
    # about testing a different format but making sure that GWCS works.

    pytest.importorskip("sunpy", minversion="6.0.1")

    data, wcs, target_wcs = aia_test_data

    output, footprint = reproject_interp((data, wcs), target_wcs, (128, 128))

    header_out = target_wcs.to_header()

    header_out["DATE-OBS"] = header_out["DATE-OBS"].replace("T", " ")

    # With sunpy 6.0.0 and later, additional keyword arguments are written out
    # so we remove these as they are not important for the comparison with the
    # reference files.
    header_out.pop("DATE-AVG", None)
    header_out.pop("MJD-AVG", None)

    return array_footprint_to_hdulist(output, footprint, header_out)


def test_reproject_roundtrip_kwarg(aia_test_data):
    # Make sure that the roundtrip_coords keyword argument has an effect. This
    # is a regression test for a bug that caused the keyword argument to be
    # ignored when in parallel/blocked mode.

    pytest.importorskip("sunpy", minversion="6.0.1")

    data, wcs, target_wcs = aia_test_data

    output_roundtrip_1 = reproject_interp(
        (data, wcs), target_wcs, shape_out=(128, 128), return_footprint=False, roundtrip_coords=True
    )
    output_roundtrip_2 = reproject_interp(
        (data, wcs),
        target_wcs,
        shape_out=(128, 128),
        return_footprint=False,
        roundtrip_coords=True,
        block_size=(32, 32),
    )

    assert_allclose(output_roundtrip_1, output_roundtrip_2)

    output_noroundtrip_1 = reproject_interp(
        (data, wcs),
        target_wcs,
        shape_out=(128, 128),
        return_footprint=False,
        roundtrip_coords=False,
    )
    output_noroundtrip_2 = reproject_interp(
        (data, wcs),
        target_wcs,
        shape_out=(128, 128),
        return_footprint=False,
        roundtrip_coords=False,
        block_size=(32, 32),
    )

    assert_allclose(output_noroundtrip_1, output_noroundtrip_2)

    # The array with round-tripping should have more NaN values:
    assert np.sum(np.isnan(output_roundtrip_1)) > 9500
    assert np.sum(np.isnan(output_noroundtrip_1)) < 7000


@pytest.mark.parametrize("roundtrip_coords", (False, True))
@pytest.mark.remote_data
def test_identity_with_offset(roundtrip_coords):
    # Reproject an array and WCS to itself but with a margin, which should
    # end up empty. This is a regression test for a bug that caused some
    # values to extend beyond the original footprint.

    wcs = WCS(naxis=2)
    wcs.wcs.ctype = "RA---TAN", "DEC--TAN"
    wcs.wcs.crpix = 322, 151
    wcs.wcs.crval = 43, 23
    wcs.wcs.cdelt = -0.1, 0.1
    wcs.wcs.equinox = 2000.0

    array_in = np.random.random((233, 123))

    wcs_out = wcs.deepcopy()
    wcs_out.wcs.crpix += 1
    shape_out = (array_in.shape[0] + 2, array_in.shape[1] + 2)

    array_out, footprint = reproject_interp(
        (array_in, wcs), wcs_out, shape_out=shape_out, roundtrip_coords=roundtrip_coords
    )

    expected = np.pad(array_in, 1, "constant", constant_values=np.nan)

    assert_allclose(expected, array_out, atol=1e-10)


def _setup_for_broadcast_test():
    with fits.open(get_pkg_data_filename("data/galactic_2d.fits", package="reproject.tests")) as pf:
        hdu_in = pf[0]
        header_in = hdu_in.header.copy()
        header_out = hdu_in.header.copy()
        header_out["CTYPE1"] = "RA---TAN"
        header_out["CTYPE2"] = "DEC--TAN"
        header_out["CRVAL1"] = 266.39311
        header_out["CRVAL2"] = -28.939779

        data = hdu_in.data

    image_stack = np.stack((data, data.T, data[::-1], data[:, ::-1]))

    # Build the reference array through un-broadcast reprojections
    array_ref = np.empty_like(image_stack)
    footprint_ref = np.empty_like(image_stack)
    for i in range(len(image_stack)):
        array_out, footprint_out = reproject_interp((image_stack[i], header_in), header_out)
        array_ref[i] = array_out
        footprint_ref[i] = footprint_out

    return image_stack, array_ref, footprint_ref, header_in, header_out


@pytest.mark.parametrize("input_extra_dims", (1, 2))
@pytest.mark.parametrize("output_shape", (None, "single", "full"))
@pytest.mark.parametrize("input_as_wcs", (True, False))
@pytest.mark.parametrize("output_as_wcs", (True, False))
def test_broadcast_reprojection(input_extra_dims, output_shape, input_as_wcs, output_as_wcs):
    image_stack, array_ref, footprint_ref, header_in, header_out = _setup_for_broadcast_test()
    # Test both single and multiple dimensions being broadcast
    if input_extra_dims == 2:
        image_stack = image_stack.reshape((2, 2, *image_stack.shape[-2:]))
        array_ref.shape = image_stack.shape
        footprint_ref.shape = image_stack.shape

    # Test different ways of providing the output shape
    if output_shape == "single":
        # Have the broadcast dimensions be auto-added to the output shape
        output_shape = image_stack.shape[-2:]
    elif output_shape == "full":
        # Provide the broadcast dimensions as part of the output shape
        output_shape = image_stack.shape

    # Ensure logic works with WCS inputs as well as Header inputs
    if input_as_wcs:
        header_in = WCS(header_in)
    if output_as_wcs:
        header_out = WCS(header_out)
        if output_shape is None:
            # This combination of parameter values is not valid
            return

    array_broadcast, footprint_broadcast = reproject_interp(
        (image_stack, header_in),
        header_out,
        output_shape,
    )

    np.testing.assert_array_equal(footprint_broadcast, footprint_ref)
    np.testing.assert_allclose(array_broadcast, array_ref)


# In the tests below we ignore FITSFixedWarning due to:
# https://github.com/astropy/astropy/pull/12844


@pytest.mark.parametrize("input_extra_dims", (1, 2))
@pytest.mark.parametrize("output_shape", (None, "single", "full"))
@pytest.mark.parametrize("parallel", [True, False])
@pytest.mark.parametrize("header_or_wcs", (lambda x: x, WCS))
@pytest.mark.filterwarnings("ignore::astropy.wcs.wcs.FITSFixedWarning")
def test_blocked_broadcast_reprojection(input_extra_dims, output_shape, parallel, header_or_wcs):
    image_stack, array_ref, footprint_ref, header_in, header_out = _setup_for_broadcast_test()
    # Test both single and multiple dimensions being broadcast
    if input_extra_dims == 2:
        image_stack = image_stack.reshape((2, 2, *image_stack.shape[-2:]))
        array_ref.shape = image_stack.shape
        footprint_ref.shape = image_stack.shape

    # Test different ways of providing the output shape
    if output_shape == "single":
        # Have the broadcast dimensions be auto-added to the output shape
        output_shape = image_stack.shape[-2:]
    elif output_shape == "full":
        # Provide the broadcast dimensions as part of the output shape
        output_shape = image_stack.shape

    # test different behavior when the output projection is a WCS
    header_out = header_or_wcs(header_out)

    array_broadcast, footprint_broadcast = reproject_interp(
        (image_stack, header_in), header_out, output_shape, parallel=parallel, block_size=[5, 5]
    )

    np.testing.assert_array_equal(footprint_broadcast, footprint_ref)
    np.testing.assert_allclose(array_broadcast, array_ref)


@pytest.mark.parametrize("parallel", [True, 2, False])
@pytest.mark.parametrize("block_size", [[500, 500], [500, 100], None])
@pytest.mark.parametrize("return_footprint", [False, True])
@pytest.mark.parametrize("existing_outputs", [False, True])
@pytest.mark.parametrize("header_or_wcs", (lambda x: x, WCS))
@pytest.mark.remote_data
@pytest.mark.filterwarnings("ignore::astropy.wcs.wcs.FITSFixedWarning")
def test_blocked_against_single(
    parallel, block_size, return_footprint, existing_outputs, header_or_wcs
):
    # Ensure when we break a reprojection down into multiple discrete blocks
    # it has the same result as if all pixels where reprejcted at once

    hdu1 = fits.open(get_pkg_data_filename("galactic_center/gc_2mass_k.fits"))[0]
    hdu2 = fits.open(get_pkg_data_filename("galactic_center/gc_msx_e.fits"))[0]
    array_test = None
    footprint_test = None

    shape_out = (720, 721)

    if existing_outputs:
        output_array_test = np.zeros(shape_out)
        output_footprint_test = np.zeros(shape_out)
        output_array_reference = np.zeros(shape_out)
        output_footprint_reference = np.zeros(shape_out)
    else:
        output_array_test = None
        output_footprint_test = None
        output_array_reference = None
        output_footprint_reference = None

    result_test = reproject_interp(
        hdu2,
        header_or_wcs(hdu1.header),
        parallel=parallel,
        block_size=block_size,
        return_footprint=return_footprint,
        output_array=output_array_test,
        output_footprint=output_footprint_test,
    )

    result_reference = reproject_interp(
        hdu2,
        header_or_wcs(hdu1.header),
        parallel=False,
        block_size=None,
        return_footprint=return_footprint,
        output_array=output_array_reference,
        output_footprint=output_footprint_reference,
    )

    if return_footprint:
        array_test, footprint_test = result_test
        array_reference, footprint_reference = result_reference
    else:
        array_test = result_test
        array_reference = result_reference

    if existing_outputs:
        assert array_test is output_array_test
        assert array_reference is output_array_reference
        if return_footprint:
            assert footprint_test is output_footprint_test
            assert footprint_reference is output_footprint_reference

    np.testing.assert_allclose(array_test, array_reference, equal_nan=True)
    if return_footprint:
        np.testing.assert_allclose(footprint_test, footprint_reference, equal_nan=True)


def test_interp_input_output_types(valid_celestial_input_data, valid_celestial_output_projections):
    # Check that all valid input/output types work properly

    array_ref, wcs_in_ref, input_value, kwargs_in = valid_celestial_input_data

    wcs_out_ref, shape_ref, output_value, kwargs_out = valid_celestial_output_projections

    # Compute reference

    output_ref, footprint_ref = reproject_interp(
        (array_ref, wcs_in_ref), wcs_out_ref, shape_out=shape_ref
    )

    # Compute test

    output_test, footprint_test = reproject_interp(
        input_value, output_value, **kwargs_in, **kwargs_out
    )

    assert_allclose(output_ref, output_test)
    assert_allclose(footprint_ref, footprint_test)


@pytest.mark.parametrize("block_size", [None, (32, 32)])
def test_reproject_order(block_size):
    # Check that the order keyword argument has an effect. This is a regression
    # test for a bug that caused the order= keyword argument to be ignored when
    # in parallel/blocked reprojection.

    with fits.open(get_pkg_data_filename("data/galactic_2d.fits", package="reproject.tests")) as pf:
        hdu_in = pf[0]

        header_out = hdu_in.header.copy()
        header_out["CTYPE1"] = "RA---TAN"
        header_out["CTYPE2"] = "DEC--TAN"
        header_out["CRVAL1"] = 266.39311
        header_out["CRVAL2"] = -28.939779

        array_out_bilinear = reproject_interp(
            hdu_in,
            header_out,
            return_footprint=False,
            order="bilinear",
            block_size=block_size,
        )

        array_out_biquadratic = reproject_interp(
            hdu_in,
            header_out,
            return_footprint=False,
            order="biquadratic",
            block_size=block_size,
        )

        with pytest.raises(AssertionError):
            assert_allclose(array_out_bilinear, array_out_biquadratic)


def test_reproject_block_size_broadcasting():
    # Regression test for a bug that caused the default chunk size to be
    # inadequate when using broadcasting in parallel mode

    array_in = np.ones((350, 250, 150))
    wcs_in = WCS(naxis=2)
    wcs_out = WCS(naxis=2)

    reproject_interp(
        (array_in, wcs_in),
        wcs_out,
        shape_out=(300, 300),
        parallel=1,
        return_footprint=False,
    )

    # Specifying a block size that is missing the extra dimension should work fine:

    reproject_interp(
        (array_in, wcs_in),
        wcs_out,
        shape_out=(300, 300),
        parallel=1,
        return_footprint=False,
        block_size=(100, 100),
    )

    # Specifying a block size with the extra dimension should work provided it matches the final output shape

    reproject_interp(
        (array_in, wcs_in),
        wcs_out,
        shape_out=(300, 300),
        parallel=1,
        return_footprint=False,
        block_size=(350, 100, 100),
    )

    # But it should fail if we specify a block size that is smaller that the total array shape

    with pytest.raises(ValueError, match="block shape should either match output data shape"):
        reproject_interp(
            (array_in, wcs_in),
            wcs_out,
            shape_out=(300, 300),
            parallel=1,
            return_footprint=False,
            block_size=(100, 100, 100),
        )


def test_reproject_dask_return_type(dask_method):
    # Regression test for a bug that caused dask arrays to not be computable
    # when using return_type='dask' when the input was a dask array.

    array_in = da.ones((35, 250, 150))
    wcs_in = WCS(naxis=2)
    wcs_out = WCS(naxis=2)

    result_numpy = reproject_interp(
        (array_in, wcs_in),
        wcs_out,
        shape_out=(300, 300),
        return_type="numpy",
        return_footprint=False,
        dask_method=dask_method,
    )

    result_dask = reproject_interp(
        (array_in, wcs_in),
        wcs_out,
        shape_out=(300, 300),
        block_size=(100, 100),
        return_type="dask",
        return_footprint=False,
        dask_method=dask_method,
    )

    assert_allclose(result_numpy, result_dask.compute(scheduler="synchronous"))


def test_auto_block_size(dask_method):
    # Unit test to make sure that specifying block_size='auto' works

    array_in = da.ones((350, 250, 150))
    wcs_in = WCS(naxis=2)
    wcs_out = WCS(naxis=2)

    # When block size and parallel aren't specified, can't return as dask arrays
    with pytest.raises(ValueError, match="Output cannot be returned as dask arrays"):
        reproject_interp(
            (array_in, wcs_in),
            wcs_out,
            shape_out=(300, 300),
            return_type="dask",
            dask_method=dask_method,
        )

    array_out, footprint_out = reproject_interp(
        (array_in, wcs_in),
        wcs_out,
        shape_out=(300, 300),
        return_type="dask",
        block_size="auto",
        dask_method=dask_method,
    )

    assert array_out.chunksize[0] == 350
    assert footprint_out.chunksize[0] == 350


@pytest.mark.parametrize("itemsize", (4, 8))
def test_bigendian_dask(itemsize, dask_method):

    # Regression test for an endianness issue that occurred when the input was
    # passed in as (dask_array, wcs) and the dask array was big endian.

    array_in_le = da.ones((35, 250, 150), dtype=f">f{itemsize}")
    array_in_be = da.ones((35, 250, 150), dtype=f"<f{itemsize}")
    wcs_in = WCS(naxis=2)
    wcs_out = WCS(naxis=2)

    array_out_be, _ = reproject_interp(
        (array_in_be, wcs_in),
        wcs_out,
        shape_out=(300, 300),
        block_size=(100, 100),
        dask_method=dask_method,
    )

    array_out_le, _ = reproject_interp(
        (array_in_le, wcs_in),
        wcs_out,
        shape_out=(300, 300),
        block_size=(100, 100),
        dask_method=dask_method,
    )

    assert_allclose(array_out_be, array_out_le)


def test_reproject_parallel_broadcasting(caplog, dask_method):

    # Unit test for reprojecting using parallelization along broadcasted
    # dimensions

    array_in = np.ones((350, 250, 150))
    wcs_in = WCS(naxis=2)
    wcs_out = WCS(naxis=2)

    # By default if we give a block size that is only in the WCS dimensions,
    # the data has a single chunk in the broadcasted dimensions

    array1 = reproject_interp(
        (array_in, wcs_in),
        wcs_out,
        shape_out=(300, 300),
        parallel=1,
        return_footprint=False,
        block_size=(100, 100),
        return_type="dask",
        dask_method=dask_method,
    )

    assert array1.chunksize == (350, 100, 100)

    assert "Broadcasting is being used" in caplog.text
    assert "Not parallelizing along broadcasted dimension" in caplog.text
    caplog.clear()

    # However, we can also have one chunk in the WCS dimensions and several in
    # the broadcasted dimensions.

    array2 = reproject_interp(
        (array_in, wcs_in),
        wcs_out,
        shape_out=(300, 300),
        parallel=1,
        return_footprint=False,
        block_size=(1, 300, 300),
        return_type="dask",
        dask_method=dask_method,
    )

    assert array2.chunksize == (1, 300, 300)

    assert "Broadcasting is being used" in caplog.text
    assert "Parallelizing along broadcasted dimension" in caplog.text
    caplog.clear()

    # However, we can also have one chunk in the WCS dimensions and several in
    # the broadcasted dimensions.

    with pytest.raises(ValueError, match="block shape should either match output"):
        reproject_interp(
            (array_in, wcs_in),
            wcs_out,
            shape_out=(300, 300),
            parallel=1,
            return_footprint=False,
            block_size=(1, 100, 100),
            return_type="dask",
            dask_method=dask_method,
        )

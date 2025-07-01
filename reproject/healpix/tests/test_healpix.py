# Licensed under a 3-clause BSD style license - see LICENSE.rst

import itertools
import os

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS
from astropy_healpix import nside_to_npix

from ...interpolation.tests.test_core import as_high_level_wcs
from ...tests.test_high_level import ALL_DTYPES
from ..high_level import reproject_from_healpix, reproject_to_healpix
from ..utils import parse_coord_system

DATA = os.path.join(os.path.dirname(__file__), "data")


def get_reference_header(overscan=1, oversample=2, nside=1):
    reference_header = fits.Header()
    reference_header.update(
        {
            "CDELT1": -180.0 / (oversample * 4 * nside),
            "CDELT2": 180.0 / (oversample * 4 * nside),
            "CRPIX1": overscan * oversample * 4 * nside,
            "CRPIX2": overscan * oversample * 2 * nside,
            "CRVAL1": 180.0,
            "CRVAL2": 0.0,
            "CTYPE1": "RA---CAR",
            "CTYPE2": "DEC--CAR",
            "CUNIT1": "deg",
            "CUNIT2": "deg",
            "NAXIS": 2,
            "NAXIS1": overscan * oversample * 8 * nside,
            "NAXIS2": overscan * oversample * 4 * nside,
        }
    )

    return reference_header


@pytest.mark.parametrize(
    "nside,nested,healpix_system,image_system,dtype,order",
    itertools.product(
        [1, 2, 4, 8, 16, 32, 64],
        [True, False],
        "C",
        "C",
        ALL_DTYPES,
        ["bilinear", "nearest-neighbor"],
    ),
)
def test_reproject_healpix_to_image_footprint(
    nside, nested, healpix_system, image_system, dtype, order
):
    """Test that HEALPix->WCS conversion correctly flags pixels that do not
    have valid WCS coordinates."""

    npix = nside_to_npix(nside)
    healpix_data = np.random.uniform(size=npix).astype(dtype)

    reference_header = get_reference_header(overscan=2, oversample=2, nside=nside)

    wcs_out = WCS(reference_header)
    shape_out = reference_header["NAXIS2"], reference_header["NAXIS1"]

    image_data, footprint = reproject_from_healpix(
        (healpix_data, healpix_system),
        wcs_out,
        shape_out=shape_out,
        order=order,
        nested=nested,
    )

    if order == "bilinear":
        expected_footprint = ~np.isnan(image_data)
    else:
        coord_system_in = parse_coord_system(healpix_system)
        yinds, xinds = np.indices(shape_out)
        world_in = wcs_out.pixel_to_world(xinds, yinds).transform_to(coord_system_in)
        world_in_unitsph = world_in.represent_as("unitspherical")
        lon_in, lat_in = world_in_unitsph.lon, world_in_unitsph.lat
        expected_footprint = ~(np.isnan(lon_in) | np.isnan(lat_in))

    np.testing.assert_array_equal(footprint, expected_footprint)


@pytest.mark.parametrize(
    "wcsapi,nside,nested,healpix_system,image_system,dtype",
    itertools.product([True, False], [1, 2, 4, 8, 16, 32, 64], [True, False], "C", "C", ALL_DTYPES),
)
def test_reproject_healpix_to_image_round_trip(
    wcsapi, nside, nested, healpix_system, image_system, dtype
):
    """Test round-trip HEALPix->WCS->HEALPix conversion for a random map,
    with a WCS projection large enough to store each HEALPix pixel"""

    npix = nside_to_npix(nside)
    healpix_data = np.random.uniform(size=npix).astype(dtype)

    reference_header = get_reference_header(oversample=2, nside=nside)

    wcs_out = WCS(reference_header)
    shape_out = reference_header["NAXIS2"], reference_header["NAXIS1"]

    if wcsapi:
        wcs_out = as_high_level_wcs(wcs_out)

    image_data, footprint = reproject_from_healpix(
        (healpix_data, healpix_system),
        wcs_out,
        shape_out=shape_out,
        order="nearest-neighbor",
        nested=nested,
    )

    healpix_data_2, footprint = reproject_to_healpix(
        (image_data, wcs_out), healpix_system, nside=nside, order="nearest-neighbor", nested=nested
    )

    np.testing.assert_array_equal(healpix_data, healpix_data_2)


def test_reproject_file():
    reference_header = get_reference_header(oversample=2, nside=8)
    data, footprint = reproject_from_healpix(
        os.path.join(DATA, "bayestar.fits.gz"), reference_header
    )
    reference_result = fits.getdata(os.path.join(DATA, "reference_result.fits"))
    np.testing.assert_allclose(data, reference_result, rtol=1.0e-5)


def test_reproject_invalid_order():
    reference_header = get_reference_header(oversample=2, nside=8)
    with pytest.raises(ValueError) as exc:
        reproject_from_healpix(
            os.path.join(DATA, "bayestar.fits.gz"), reference_header, order="bicubic"
        )
    assert exc.value.args[0] == "Only nearest-neighbor and bilinear interpolation are supported"


def test_reproject_to_healpix_input_types(valid_celestial_input_data):
    array_ref, wcs_in_ref, input_value, kwargs_in = valid_celestial_input_data

    # For now we don't support 3D arrays in reproject_to_healpix
    if array_ref.ndim == 3:
        pytest.skip()

    # Compute reference

    healpix_data_ref, footprint_ref = reproject_to_healpix((array_ref, wcs_in_ref), "C", nside=64)

    # Compute test

    healpix_data_test, footprint_test = reproject_to_healpix(
        input_value, "C", nside=64, **kwargs_in
    )

    # Make sure there are some valid values

    assert np.sum(~np.isnan(healpix_data_ref)) == 4

    np.testing.assert_allclose(healpix_data_ref, healpix_data_test)
    np.testing.assert_allclose(footprint_ref, footprint_test)


def test_reproject_from_healpix_output_types(valid_celestial_output_projections):
    wcs_out_ref, shape_ref, output_value, kwargs_out = valid_celestial_output_projections

    array_input = np.random.random(12 * 64**2)

    # Compute reference

    output_ref, footprint_ref = reproject_from_healpix(
        (array_input, "C"), wcs_out_ref, nested=True, shape_out=shape_ref
    )

    # Compute test

    output_test, footprint_test = reproject_from_healpix(
        (array_input, "C"), output_value, nested=True, **kwargs_out
    )

    np.testing.assert_allclose(output_ref, output_test)
    np.testing.assert_allclose(footprint_ref, footprint_test)


def test_reproject_to_healpix_exact_allsky():

    # Regression test for a bug that caused artifacts in the final image if the
    # WCS covered the whole sky - this was due to using scipy's map_coordinates
    # one instead of our built-in one which deals properly with the pixels
    # around the rim.

    shape_out = (160, 320)
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [(shape_out[1] + 1) / 2, (shape_out[0] + 1) / 2]
    wcs.wcs.cdelt = np.array([-360.0 / shape_out[1], 180.0 / shape_out[0]])
    wcs.wcs.crval = [0, 0]
    wcs.wcs.ctype = ["RA---CAR", "DEC--CAR"]

    array = np.ones(shape_out)

    healpix_array, footprint = reproject_to_healpix(
        (array, wcs),
        coord_system_out="galactic",
        nside=64,
        nested=False,
        order="bilinear",
    )

    assert np.all(footprint > 0)
    assert not np.any(np.isnan(healpix_array))

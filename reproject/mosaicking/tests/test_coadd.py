# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import random

import numpy as np
import pytest
from astropy.io import fits
from astropy.io.fits import Header
from astropy.wcs import WCS
from numpy.testing import assert_allclose

from reproject import reproject_exact, reproject_interp
from reproject.mosaicking.coadd import reproject_and_coadd
from reproject.tests.helpers import array_footprint_to_hdulist

ATOL = 1.0e-9

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "tests", "data")


@pytest.fixture(params=[reproject_interp, reproject_exact], ids=["interp", "exact"])
def reproject_function(request):
    return request.param


@pytest.fixture(params=[False, True])
def intermediate_memmap(request):
    return request.param


class TestReprojectAndCoAdd:
    def setup_method(self, method):
        self.wcs = WCS(naxis=2)
        self.wcs.wcs.ctype = "RA---TAN", "DEC--TAN"
        self.wcs.wcs.crpix = 322, 151
        self.wcs.wcs.crval = 43, 23
        self.wcs.wcs.cdelt = -0.1, 0.1
        self.wcs.wcs.equinox = 2000.0

        self.array = np.random.random((399, 334))

    def _get_tiles(self, views):
        # Given a list of views as (imin, imax, jmin, jmax), construct
        #  tiles that can be passed into the co-adding code

        input_data = []

        for view in views:
            array = self.array[view].copy()
            wcs = self.wcs.deepcopy()
            wcs = wcs[view]
            input_data.append((array, wcs))

        return input_data

    @property
    def _nonoverlapping_views(self):
        ie = (0, 122, 233, 245, 334)
        je = (0, 44, 45, 333, 335, 399)

        views = []
        for i in range(4):
            for j in range(5):
                views.append(np.s_[je[j] : je[j + 1], ie[i] : ie[i + 1]])

        return views

    @property
    def _overlapping_views(self):
        ie = (0, 122, 233, 245, 334)
        je = (0, 44, 45, 333, 335, 399)

        views = []
        for i in range(4):
            for j in range(5):
                views.append(np.s_[je[j] : je[j + 1] + 10, ie[i] : ie[i + 1] + 10])

        return views

    @pytest.mark.parametrize("combine_function", ["mean", "sum", "first", "last"])
    def test_coadd_no_overlap(self, combine_function, reproject_function, intermediate_memmap):
        # Make sure that if all tiles are exactly non-overlapping, and
        # we use 'sum' or 'mean', we get the exact input array back.

        input_data = self._get_tiles(self._nonoverlapping_views)

        array, footprint = reproject_and_coadd(
            input_data,
            self.wcs,
            shape_out=self.array.shape,
            combine_function=combine_function,
            reproject_function=reproject_function,
        )

        assert_allclose(array, self.array, atol=ATOL)
        assert_allclose(footprint, 1, atol=ATOL)

    def test_coadd_with_overlap(self, reproject_function, intermediate_memmap):
        # Here we make the input tiles overlapping. We can only check the
        # mean, not the sum.

        input_data = self._get_tiles(self._overlapping_views)

        array, footprint = reproject_and_coadd(
            input_data,
            self.wcs,
            shape_out=self.array.shape,
            combine_function="mean",
            reproject_function=reproject_function,
        )

        assert_allclose(array, self.array, atol=ATOL)

    def test_coadd_with_outputs(self, tmp_path, reproject_function, intermediate_memmap):
        # Test the options to specify output array/footprint

        input_data = self._get_tiles(self._overlapping_views)

        output_array = np.memmap(
            tmp_path / "array.np", mode="w+", dtype=float, shape=self.array.shape
        )
        output_footprint = np.memmap(
            tmp_path / "footprint.np", mode="w+", dtype=float, shape=self.array.shape
        )

        array, footprint = reproject_and_coadd(
            input_data,
            self.wcs,
            shape_out=self.array.shape,
            combine_function="mean",
            reproject_function=reproject_function,
            output_array=output_array,
            output_footprint=output_footprint,
        )

        assert_allclose(output_array, self.array, atol=ATOL)
        assert_allclose(output_footprint, footprint, atol=ATOL)

    @pytest.mark.parametrize("combine_function", ["first", "last", "min", "max"])
    def test_coadd_with_overlap_first_last(self, reproject_function, combine_function):
        views = self._overlapping_views
        input_data = self._get_tiles(views)

        # Make each of the overlapping tiles different
        for i, (array, wcs) in enumerate(input_data):
            # We give each tile integer values that range from 0 to 19 but we
            # deliberately don't make the first one 0 and the last one 19 so
            # that min/max differs from first/last.
            input_data[i] = (np.full_like(array, (i + 7) % 20), wcs)

        array, footprint = reproject_and_coadd(
            input_data,
            self.wcs,
            shape_out=self.array.shape,
            combine_function=combine_function,
            reproject_function=reproject_function,
        )

        # Test that either the correct tile sets the output value in the overlap regions
        test_sequence = list(enumerate(views))

        if combine_function == "last":
            test_sequence = test_sequence[::-1]
        elif combine_function == "min":
            test_sequence = test_sequence[13:] + test_sequence[:13]
        elif combine_function == "max":
            test_sequence = (test_sequence[13:] + test_sequence[:13])[::-1]

        for i, view in test_sequence:
            # Each tile in test_sequence should overwrite the following tiles
            # in the overlap regions. We'll use NaNs to mark pixels in the
            # output array that have already been set by a preceding tile, so
            # we'll go through, check that each tile matches the non-nan pixels
            # in its region, and then set that whole region to nan.
            output_tile = array[view]
            output_values = output_tile[np.isfinite(output_tile)]
            assert_allclose(output_values, (i + 7) % 20)
            array[view] = np.nan

    def test_coadd_background_matching(self, reproject_function, intermediate_memmap):
        # Test out the background matching

        input_data = self._get_tiles(self._overlapping_views)

        for array, _ in input_data:
            array += random.uniform(-3, 3)

        # First check that without background matching the arrays don't match

        array, footprint = reproject_and_coadd(
            input_data,
            self.wcs,
            shape_out=self.array.shape,
            combine_function="mean",
            reproject_function=reproject_function,
        )

        assert not np.allclose(array, self.array, atol=ATOL)

        # Now check that once the backgrounds are matched the values agree

        array, footprint = reproject_and_coadd(
            input_data,
            self.wcs,
            shape_out=self.array.shape,
            combine_function="mean",
            reproject_function=reproject_function,
            match_background=True,
        )

        # The absolute values of the two arrays will be offset since any
        # solution that reproduces the offsets between images is valid

        assert_allclose(array - np.mean(array), self.array - np.mean(self.array), atol=ATOL)

    def test_coadd_background_matching_one_array(self, reproject_function, intermediate_memmap):
        # Test that background matching doesn't affect the output when there's
        # only one input image.

        input_data = [(self.array, self.wcs)]

        array_matched, footprint_matched = reproject_and_coadd(
            input_data,
            self.wcs,
            shape_out=self.array.shape,
            combine_function="mean",
            reproject_function=reproject_function,
            match_background=True,
        )

        array, footprint = reproject_and_coadd(
            input_data,
            self.wcs,
            shape_out=self.array.shape,
            combine_function="mean",
            reproject_function=reproject_function,
            match_background=False,
        )
        np.testing.assert_allclose(array, array_matched)
        np.testing.assert_allclose(footprint, footprint_matched)

    @pytest.mark.parametrize("combine_function", ["first", "last", "min", "max", "sum", "mean"])
    @pytest.mark.parametrize("match_background", [True, False])
    def test_footprint_correct(self, reproject_function, combine_function, match_background):
        # Test that the output array is zero outside the returned footprint
        # We're running this test over a somewhat large grid of parameters, so
        # cut down the array size to avoid increasing the total test runtime
        # too much.
        slice = np.s_[::3, ::3]
        wcs1 = self.wcs[slice]
        wcs2 = self.wcs.deepcopy()
        # Add a 45-degree rotation
        wcs2.wcs.pc = np.array([[0.5, -0.5], [0.5, 0.5]])

        wcs_out = wcs1.deepcopy()
        # Expand the output WCS to go beyond the input images
        wcs_out.wcs.cdelt = 2 * wcs1.wcs.cdelt[0], 2 * wcs1.wcs.cdelt[1]

        # Ensure the input data is fully non-zero, so we can tell where data
        # got projected to in the output image.
        array1 = np.full_like(self.array[slice], 2)
        array2 = array1 + 5

        array, footprint = reproject_and_coadd(
            [(array1, wcs1), (array2, wcs2)],
            wcs_out,
            shape_out=self.array.shape,
            combine_function=combine_function,
            reproject_function=reproject_function,
            match_background=match_background,
        )

        assert np.all((array != 0) == (footprint > 0))

    def test_coadd_background_matching_with_nan(self, reproject_function, intermediate_memmap):
        # Test out the background matching when NaN values are present. We do
        # this by using three arrays with the same footprint but with different
        # parts masked.

        array1 = self.array.copy() + random.uniform(-3, 3)
        array2 = self.array.copy() + random.uniform(-3, 3)
        array3 = self.array.copy() + random.uniform(-3, 3)

        array1[:, 122:] = np.nan
        array2[:, :50] = np.nan
        array2[:, 266:] = np.nan
        array3[:, :199] = np.nan

        input_data = [(array1, self.wcs), (array2, self.wcs), (array3, self.wcs)]

        array, footprint = reproject_and_coadd(
            input_data,
            self.wcs,
            shape_out=self.array.shape,
            combine_function="mean",
            reproject_function=reproject_function,
            match_background=True,
        )

        # The absolute values of the two arrays will be offset since any
        # solution that reproduces the offsets between images is valid

        assert_allclose(array - np.mean(array), self.array - np.mean(self.array), atol=ATOL)

    @pytest.mark.filterwarnings("ignore:unclosed file:ResourceWarning")
    @pytest.mark.parametrize("mode", ["arrays", "filenames", "hdus", "hdulist"])
    def test_coadd_with_weights(self, tmpdir, reproject_function, mode, intermediate_memmap):
        # Make sure that things work properly when specifying weights

        array1 = self.array + 1
        array2 = self.array - 1

        weight1 = np.cumsum(np.ones_like(self.array), axis=1) - 1
        weight2 = weight1[:, ::-1]

        input_data = [(array1, self.wcs), (array2, self.wcs)]

        if mode == "arrays":
            input_weights = [weight1, weight2]
        elif mode == "filenames":
            filename1 = tmpdir.join("weight1.fits").strpath
            filename2 = tmpdir.join("weight2.fits").strpath
            fits.writeto(filename1, weight1)
            fits.writeto(filename2, weight2)
            input_weights = [filename1, filename2]
        elif mode == "hdus":
            hdu1 = fits.ImageHDU(weight1)
            hdu2 = fits.ImageHDU(weight2)
            input_weights = [hdu1, hdu2]
        elif mode == "hdulist":
            hdu1 = fits.HDUList([fits.ImageHDU(weight1, header=self.wcs.to_header())])
            hdu2 = fits.HDUList([fits.ImageHDU(weight2, header=self.wcs.to_header())])
            input_weights = [hdu1, hdu2]

        array, footprint = reproject_and_coadd(
            input_data,
            self.wcs,
            shape_out=self.array.shape,
            combine_function="mean",
            input_weights=input_weights,
            reproject_function=reproject_function,
            match_background=False,
        )

        expected = self.array + (2 * (weight1 / weight1.max()) - 1)

        assert_allclose(array, expected, atol=ATOL)

    @pytest.mark.filterwarnings("ignore:unclosed file:ResourceWarning")
    def test_coadd_with_weights_with_wcs(self, tmpdir, reproject_function, intermediate_memmap):
        # Make sure that things work properly when specifying weights that have offset WCS

        array1 = self.array + 1
        array2 = self.array - 1

        weight1 = np.cumsum(np.ones_like(self.array), axis=1) - 1
        weight2 = weight1[:, ::-1]

        input_data = [(array1, self.wcs), (array2, self.wcs)]

        # make weight WCS pixel scale bigger so that weights encompass data
        weightwcs = self.wcs.copy()
        weightwcs.wcs.cdelt *= 1.1

        hdu1 = fits.ImageHDU(weight1, header=weightwcs.to_header())
        hdu2 = fits.ImageHDU(weight2, header=weightwcs.to_header())
        input_weights = [hdu1, hdu2]

        array, footprint = reproject_and_coadd(
            input_data,
            self.wcs,
            shape_out=self.array.shape,
            combine_function="mean",
            input_weights=input_weights,
            reproject_function=reproject_function,
            match_background=False,
        )

        weights1_reprojected = reproject_function(
            hdu1, self.wcs, shape_out=self.array.shape, return_footprint=False
        )
        weights2_reprojected = reproject_function(
            hdu2, self.wcs, shape_out=self.array.shape, return_footprint=False
        )
        array1_reprojected = reproject_function(
            input_data[0], self.wcs, shape_out=self.array.shape, return_footprint=False
        )
        array2_reprojected = reproject_function(
            input_data[1], self.wcs, shape_out=self.array.shape, return_footprint=False
        )
        expected = (
            array1_reprojected * weights1_reprojected + array2_reprojected * weights2_reprojected
        ) / (weights1_reprojected + weights2_reprojected)

        assert_allclose(array, expected, atol=ATOL)

    @pytest.mark.parametrize("block_size_mode", (None, "block_size", "block_sizes"))
    def test_coadd_with_broadcasting(
        self, reproject_function, intermediate_memmap, block_size_mode
    ):

        # Coadding should work with broadcasting, i.e. the fact the
        # input/output WCS might have fewer dimensions than the data.

        input_data = self._get_tiles(self._overlapping_views)

        input_data = [
            (np.broadcast_to(array.reshape((1,) + array.shape), (3,) + array.shape), wcs)
            for array, wcs in input_data
        ]

        if block_size_mode == "block_size":
            kwargs = {"block_size": (1,) + self.array.shape}
        elif block_size_mode == "block_size":
            kwargs = {"block_sizes": (1,) + self.array.shape}
        else:
            kwargs = {}

        array, footprint = reproject_and_coadd(
            input_data,
            self.wcs,
            shape_out=(3,) + self.array.shape,
            combine_function="mean",
            reproject_function=reproject_function,
            **kwargs,
        )

        for index in range(3):
            assert_allclose(array[index], self.array, atol=ATOL)


HEADER_SOLAR_OUT = """
WCSAXES =                    2
CRPIX1  =                 90.5
CRPIX2  =                 45.5
CDELT1  =                    2
CDELT2  =                    2
CUNIT1  = 'deg'
CUNIT2  = 'deg'
CTYPE1  = 'HGLN-CAR'
CTYPE2  = 'HGLT-CAR'
CRVAL1  =                  0.0
CRVAL2  =                  0.0
LONPOLE =                  0.0
LATPOLE =                 90.0
DATE-OBS= '2011-02-15T00:14:03.654'
MJD-OBS =      55607.009764514
MJD-OBS =      55607.009764514
"""


@pytest.mark.array_compare()
def test_coadd_solar_map():
    # This is a test that exercises a lot of different parts of the mosaicking
    # code. The idea is to take three solar images from different viewpoints
    # and combine them into a single one. This uses weight maps that are not
    # uniform and also include NaN values.

    pytest.importorskip("sunpy", minversion="6.0.1")
    from sunpy.map import Map, all_coordinates_from_map

    # Load in three images from different viewpoints around the Sun
    filenames = ["secchi_l0_a.fits", "aia_171_level1.fits", "secchi_l0_b.fits"]
    maps = [Map(os.path.join(DATA, f)) for f in filenames]

    # Produce weight maps that are centered on the solar disk and go to zero at the edges
    coordinates = tuple(map(all_coordinates_from_map, maps))
    input_weights = [coord.transform_to("heliocentric").z.value for coord in coordinates]
    input_weights = [(w / np.nanmax(w)) ** 4 for w in input_weights]

    shape_out = [90, 180]
    wcs_out = WCS(Header.fromstring(HEADER_SOLAR_OUT, sep="\n"))
    scales = [1 / 6, 1, 1 / 6]

    input_data = tuple((a.data * scale, a.wcs) for (a, scale) in zip(maps, scales, strict=True))

    array, footprint = reproject_and_coadd(
        input_data,
        wcs_out,
        shape_out,
        input_weights=input_weights,
        reproject_function=reproject_interp,
        match_background=True,
    )

    header_out = wcs_out.to_header()

    return array_footprint_to_hdulist(array, footprint, header_out)

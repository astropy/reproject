# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import random
import warnings

import dask.array as da
import numpy as np
import pytest
from astropy.io import fits
from astropy.io.fits import Header
from astropy.wcs import WCS
from numpy.testing import assert_allclose

from reproject import reproject_exact, reproject_interp
from reproject.mosaicking._coadd import reproject_and_coadd
from reproject.tests.helpers import array_footprint_to_hdulist
from reproject.tests.test_non_reprojected_dims import _drifting_cube_wcs

ATOL = 1.0e-9

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "tests", "data")


@pytest.fixture(params=[reproject_interp, reproject_exact], ids=["interp", "exact"])
def reproject_function(request):
    return request.param


@pytest.fixture(params=[False, True, "zarr"])
def intermediate_memmap(request):
    return request.param


@pytest.fixture(params=[None, "dask"], ids=["numpy", "dask"])
def return_type(request):
    return request.param


def _compute(result, return_type):
    # For return_type='dask' the co-addition must hand back an uncomputed dask array
    # (a silent numpy fallback would defeat the point), so assert that before computing
    # it to plain numpy for the numerical comparisons in the tests below.
    if return_type == "dask":
        assert isinstance(result, da.core.Array)
    return np.asarray(result)


@pytest.fixture(params=[False, True])
def intermediate_memmap_nozarr(request):
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
    def test_coadd_no_overlap(
        self, combine_function, reproject_function, intermediate_memmap, return_type
    ):
        # Make sure that if all tiles are exactly non-overlapping, and
        # we use 'sum' or 'mean', we get the exact input array back.

        if return_type == "dask" and intermediate_memmap:
            pytest.skip("return_type='dask' does not support intermediate_memmap")

        input_data = self._get_tiles(self._nonoverlapping_views)

        array, footprint = reproject_and_coadd(
            input_data,
            self.wcs,
            shape_out=self.array.shape,
            combine_function=combine_function,
            reproject_function=reproject_function,
            return_type=return_type,
            intermediate_memmap=intermediate_memmap,
        )
        # np.asarray is a no-op for the numpy path and computes the deferred dask
        # result for return_type='dask', which must match numerically.
        array = _compute(array, return_type)
        footprint = _compute(footprint, return_type)

        assert_allclose(array, self.array, atol=ATOL)
        assert_allclose(footprint, 1, atol=ATOL)

    def test_coadd_with_overlap(self, reproject_function, intermediate_memmap, return_type):
        # Here we make the input tiles overlapping. We can only check the
        # mean, not the sum.

        if return_type == "dask" and intermediate_memmap:
            pytest.skip("return_type='dask' does not support intermediate_memmap")

        input_data = self._get_tiles(self._overlapping_views)

        array, footprint = reproject_and_coadd(
            input_data,
            self.wcs,
            shape_out=self.array.shape,
            combine_function="mean",
            reproject_function=reproject_function,
            return_type=return_type,
            intermediate_memmap=intermediate_memmap,
        )
        array = _compute(array, return_type)

        assert_allclose(array, self.array, atol=ATOL)

    def test_coadd_dask_median(self, reproject_function):
        # Check the deferred median against a direct numpy nanmedian of the
        # reprojected images, and that the result is uncomputed.
        input_data = self._get_tiles(self._overlapping_views)

        array, footprint = reproject_and_coadd(
            input_data,
            self.wcs,
            shape_out=self.array.shape,
            combine_function="median",
            reproject_function=reproject_function,
            return_type="dask",
        )
        assert isinstance(array, da.core.Array)  # returned uncomputed

        refs = []
        for arr, wcs in input_data:
            reprojected, fp = reproject_function((arr, wcs), self.wcs, shape_out=self.array.shape)
            reprojected[fp == 0] = np.nan
            refs.append(reprojected)
        reference = np.nanmedian(np.array(refs), axis=0)

        covered = np.asarray(footprint) > 0
        assert_allclose(np.asarray(array)[covered], reference[covered], atol=ATOL)

        # The return_type='numpy' path combines the retained reprojected
        # arrays chunk by chunk and must match the deferred median exactly
        array_numpy, footprint_numpy = reproject_and_coadd(
            input_data,
            self.wcs,
            shape_out=self.array.shape,
            combine_function="median",
            reproject_function=reproject_function,
        )
        assert_allclose(array_numpy, np.asarray(array))
        assert_allclose(footprint_numpy, np.asarray(footprint))

    @pytest.mark.parametrize("intermediate_memmap", [False, True])
    def test_coadd_numpy_median_memmap(self, reproject_function, intermediate_memmap):
        # The retained reprojected arrays can be kept on disk while computing
        # the median chunk by chunk
        input_data = self._get_tiles(self._overlapping_views)

        array, footprint = reproject_and_coadd(
            input_data,
            self.wcs,
            shape_out=self.array.shape,
            combine_function="median",
            reproject_function=reproject_function,
            intermediate_memmap=intermediate_memmap,
        )
        covered = footprint > 0
        assert_allclose(array[covered], self.array[covered], atol=ATOL)

    def test_coadd_numpy_median_match_background(self, reproject_function):
        # match_background composes with the median since the corrections are
        # applied to the retained arrays before the combine (this is not
        # possible with the deferred paths)
        input_data = self._get_tiles(self._overlapping_views)
        input_data = [(array + iview, wcs) for iview, (array, wcs) in enumerate(input_data)]

        array, footprint = reproject_and_coadd(
            input_data,
            self.wcs,
            shape_out=self.array.shape,
            combine_function="median",
            reproject_function=reproject_function,
            match_background=True,
        )
        covered = footprint > 0
        # The mean of the corrections is zero, so the matched result is offset
        # from the reference by the mean of the applied offsets
        offset = np.mean(np.arange(len(input_data)))
        assert_allclose(array[covered], (self.array + offset)[covered], atol=ATOL)

    def test_coadd_dask_median_uncovered(self):
        # Pixels covered by no image must not make the deferred median warn about
        # all-NaN slices at compute time, and must come out blank with a zero
        # footprint.
        input_data = self._get_tiles([np.s_[0:100, 0:100]])

        array, footprint = reproject_and_coadd(
            input_data,
            self.wcs,
            shape_out=self.array.shape,
            combine_function="median",
            reproject_function=reproject_interp,
            return_type="dask",
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            array = np.asarray(array)
            footprint = np.asarray(footprint)

        assert_allclose(footprint[200:, 200:], 0, atol=ATOL)
        assert_allclose(array[200:, 200:], 0, atol=ATOL)
        assert_allclose(array[:100, :100], self.array[:100, :100], atol=ATOL)

    @pytest.mark.parametrize("combine_function", ["mean", "sum"])
    def test_coadd_dask_negative_weights(self, combine_function):
        # Reprojected weights can end up negative (e.g. interpolation overshoot
        # with order='bicubic'), and the return_type='numpy' path includes negative
        # contributions in both the weighted sum and the summed footprint, and only
        # blanks pixels whose summed footprint is exactly zero. The deferred dask
        # path must match, in particular not blanking pixels whose summed footprint
        # is negative.
        views = [np.s_[0:250, :], np.s_[150:399, :]]
        input_data = self._get_tiles(views)
        input_weights = [
            np.ones_like(input_data[0][0]),
            np.full_like(input_data[1][0], -0.5),
        ]

        results = {}
        for return_type in (None, "dask"):
            array, footprint = reproject_and_coadd(
                input_data,
                self.wcs,
                shape_out=self.array.shape,
                input_weights=input_weights,
                combine_function=combine_function,
                reproject_function=reproject_interp,
                return_type=return_type,
            )
            results[return_type] = (np.asarray(array), np.asarray(footprint))

        array_numpy, footprint_numpy = results[None]
        array_dask, footprint_dask = results["dask"]

        # The lower band is covered only by the negatively-weighted image, so its
        # summed footprint is negative there and the values must be kept.
        assert footprint_numpy[300, 100] < 0
        assert_allclose(
            array_numpy[300:, :],
            self.array[300:, :] * (1 if combine_function == "mean" else -0.5),
            atol=ATOL,
        )

        assert_allclose(footprint_dask, footprint_numpy, atol=ATOL)
        assert_allclose(array_dask, array_numpy, atol=ATOL)

    def test_coadd_dask_duplicate_input_names(self):
        # Dask identifies arrays by name, so two different input dask arrays
        # sharing a name get silently deduplicated once the co-addition is
        # combined into one graph, with one input's data used for both; the
        # dask path must warn about this. Passing the same array object twice
        # is legitimate and must stay silent.
        views = [np.s_[0:200, :], np.s_[199:399, :]]
        (array1, wcs1), (array2, wcs2) = self._get_tiles(views)
        dup1 = da.from_array(array1, name="duplicate-name")
        dup2 = da.from_array(array2, name="duplicate-name")

        # Record warnings rather than using pytest.warns, which re-emits the
        # non-matching warnings on exit: with the oldest dependencies the dask
        # machinery emits an unrelated DeprecationWarning that the
        # warnings-as-errors filter would then turn into a failure.
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            reproject_and_coadd(
                [(dup1, wcs1), (dup2, wcs2)],
                self.wcs,
                shape_out=self.array.shape,
                combine_function="mean",
                reproject_function=reproject_interp,
                return_type="dask",
            )
        assert any(
            issubclass(w.category, UserWarning) and "share the name" in str(w.message)
            for w in recorded
        )

        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            reproject_and_coadd(
                [(dup1, wcs1), (dup1, wcs1)],
                self.wcs,
                shape_out=self.array.shape,
                combine_function="mean",
                reproject_function=reproject_interp,
                return_type="dask",
            )
        assert not any("share the name" in str(w.message) for w in recorded)

    @pytest.mark.parametrize(
        "combine_function", ["mean", "sum", "first", "last", "min", "max", "median"]
    )
    def test_coadd_dask_chunked_combine(self, combine_function):
        # The dask co-addition assembles each output chunk from the images that
        # overlap it; with chunks much smaller than the mosaic this exercises
        # the per-chunk placement and combination for every combine function,
        # which must match the return_type='numpy' path (or, for 'median',
        # which that path cannot compute, a direct nanmedian of the
        # individually reprojected images). The tiles are scaled to different
        # values so that the selecting combine functions are non-trivial.
        input_data = [
            (array * (index + 1), wcs)
            for index, (array, wcs) in enumerate(self._get_tiles(self._overlapping_views))
        ]

        array, footprint = reproject_and_coadd(
            input_data,
            self.wcs,
            shape_out=self.array.shape,
            combine_function=combine_function,
            reproject_function=reproject_interp,
            return_type="dask",
            block_size=(100, 100),
        )
        array = np.asarray(array)
        footprint = np.asarray(footprint)

        if combine_function == "median":
            refs = []
            for arr, wcs in input_data:
                reprojected, fp = reproject_interp((arr, wcs), self.wcs, shape_out=self.array.shape)
                reprojected[fp == 0] = np.nan
                refs.append(reprojected)
            reference = np.nanmedian(np.array(refs), axis=0)
            covered = footprint > 0
            assert covered.any()
            assert_allclose(array[covered], reference[covered], atol=ATOL)
        else:
            reference, reference_footprint = reproject_and_coadd(
                input_data,
                self.wcs,
                shape_out=self.array.shape,
                combine_function=combine_function,
                reproject_function=reproject_interp,
            )
            assert_allclose(array, reference, atol=ATOL)
            assert_allclose(footprint, reference_footprint, atol=ATOL)

    def test_coadd_dask_graph_scales_with_overlap(self):
        # Each output chunk depends only on the images that actually overlap
        # it, so no zero-padding chunks are materialized and the graph size
        # scales with the total overlap area rather than with the number of
        # images times the number of chunks (padding every image to the full
        # mosaic and reducing along a stacking axis produced several times
        # more tasks for this configuration).
        input_data = self._get_tiles(self._nonoverlapping_views)

        array, footprint = reproject_and_coadd(
            input_data,
            self.wcs,
            shape_out=self.array.shape,
            combine_function="mean",
            reproject_function=reproject_interp,
            return_type="dask",
            block_size=(100, 100),
        )
        assert len(dict(array.__dask_graph__())) < 1500

    def test_coadd_dask_rejects_unsupported(self):
        # Options that fill arrays in place cannot apply to a deferred dask result
        # and must raise rather than being silently ignored.
        input_data = self._get_tiles(self._nonoverlapping_views[:2])

        for kwargs in (
            {"output_array": np.zeros(self.array.shape)},
            {"output_footprint": np.zeros(self.array.shape)},
            {"intermediate_memmap": True},
            {"match_background": True},
        ):
            with pytest.raises(ValueError, match="Cannot use return_type='dask'"):
                reproject_and_coadd(
                    input_data,
                    self.wcs,
                    shape_out=self.array.shape,
                    combine_function="mean",
                    reproject_function=reproject_interp,
                    return_type="dask",
                    **kwargs,
                )

    def test_coadd_return_type_validated(self):
        # An unknown return_type must raise instead of silently running the eager
        # numpy co-addition, and 'numpy' is accepted as an alias for the default.
        input_data = self._get_tiles(self._nonoverlapping_views[:2])

        with pytest.raises(ValueError, match="return_type should be set to"):
            reproject_and_coadd(
                input_data,
                self.wcs,
                shape_out=self.array.shape,
                combine_function="mean",
                reproject_function=reproject_interp,
                return_type="Dask",
            )

        array, footprint = reproject_and_coadd(
            input_data,
            self.wcs,
            shape_out=self.array.shape,
            combine_function="mean",
            reproject_function=reproject_interp,
            return_type="numpy",
        )
        assert isinstance(array, np.ndarray)

    def test_coadd_no_overlap_with_output(self, return_type):
        # When no input is predicted to overlap the output, both paths must return
        # a blank mosaic and zero footprint rather than erroring.
        input_data = self._get_tiles(self._nonoverlapping_views[:2])

        wcs_out = self.wcs.deepcopy()
        wcs_out.wcs.crval = 103, 23

        array, footprint = reproject_and_coadd(
            input_data,
            wcs_out,
            shape_out=self.array.shape,
            combine_function="mean",
            reproject_function=reproject_interp,
            blank_pixel_value=-1,
            return_type=return_type,
        )
        array = _compute(array, return_type)
        footprint = _compute(footprint, return_type)

        assert_allclose(array, -1, atol=ATOL)
        assert_allclose(footprint, 0, atol=ATOL)

    def test_coadd_dask_block_size_auto(self):
        # block_size='auto' is a documented value and must work with the deferred
        # dask co-addition too.
        input_data = self._get_tiles(self._nonoverlapping_views)

        array, footprint = reproject_and_coadd(
            input_data,
            self.wcs,
            shape_out=self.array.shape,
            combine_function="mean",
            reproject_function=reproject_interp,
            return_type="dask",
            block_size="auto",
        )

        assert_allclose(np.asarray(array), self.array, atol=ATOL)

    def test_coadd_dask_output_chunking(self):
        # A single user-specified block size is used as the chunking of the
        # returned dask arrays.
        input_data = self._get_tiles(self._nonoverlapping_views)

        array, footprint = reproject_and_coadd(
            input_data,
            self.wcs,
            shape_out=self.array.shape,
            combine_function="mean",
            reproject_function=reproject_interp,
            return_type="dask",
            block_size=(100, 100),
        )

        expected = da.core.normalize_chunks((100, 100), shape=self.array.shape, dtype=float)
        assert array.chunks == expected
        assert footprint.chunks == expected

    def test_coadd_block_sizes_single_tuple_matching_n_datasets(self, return_type):
        # A single flat block size tuple whose length happens to equal the number
        # of datasets must still be interpreted as one common block size (this
        # previously raised TypeError in the per-dataset detection).
        input_data = self._get_tiles(self._nonoverlapping_views[:2])

        array, footprint = reproject_and_coadd(
            input_data,
            self.wcs,
            shape_out=self.array.shape,
            combine_function="mean",
            reproject_function=reproject_interp,
            block_sizes=(10, 10),
            return_type=return_type,
        )
        array = _compute(array, return_type)
        footprint = _compute(footprint, return_type)

        assert_allclose(array[footprint > 0], self.array[footprint > 0], atol=ATOL)

    def test_coadd_block_sizes_wrong_length(self):
        # A per-dataset list of block sizes must have one entry per dataset.
        input_data = self._get_tiles(self._nonoverlapping_views[:2])

        with pytest.raises(ValueError, match="one per dataset"):
            reproject_and_coadd(
                input_data,
                self.wcs,
                shape_out=self.array.shape,
                combine_function="mean",
                reproject_function=reproject_interp,
                block_sizes=[(10, 10), (10, 10), (10, 10)],
            )

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
            intermediate_memmap=intermediate_memmap,
        )

        assert_allclose(output_array, self.array, atol=ATOL)
        assert_allclose(output_footprint, footprint, atol=ATOL)

    @pytest.mark.parametrize("combine_function", ["first", "last", "min", "max"])
    def test_coadd_with_overlap_first_last(self, reproject_function, combine_function, return_type):
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
            return_type=return_type,
        )
        array = _compute(array, return_type)

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

    def test_coadd_background_matching(self, reproject_function, intermediate_memmap_nozarr):
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
            intermediate_memmap=intermediate_memmap_nozarr,
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
            intermediate_memmap=intermediate_memmap_nozarr,
        )

        # The absolute values of the two arrays will be offset since any
        # solution that reproduces the offsets between images is valid

        assert_allclose(array - np.mean(array), self.array - np.mean(self.array), atol=ATOL)

    def test_coadd_background_matching_one_array(
        self, reproject_function, intermediate_memmap_nozarr
    ):
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
            intermediate_memmap=intermediate_memmap_nozarr,
        )

        array, footprint = reproject_and_coadd(
            input_data,
            self.wcs,
            shape_out=self.array.shape,
            combine_function="mean",
            reproject_function=reproject_function,
            match_background=False,
            intermediate_memmap=intermediate_memmap_nozarr,
        )
        np.testing.assert_allclose(array, array_matched)
        np.testing.assert_allclose(footprint, footprint_matched)

    @pytest.mark.parametrize("combine_function", ["first", "last", "min", "max", "sum", "mean"])
    @pytest.mark.parametrize("match_background", [True, False])
    def test_footprint_correct(
        self, reproject_function, combine_function, match_background, return_type
    ):
        if return_type == "dask" and match_background:
            pytest.skip("return_type='dask' does not support match_background")
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
            return_type=return_type,
        )
        array = _compute(array, return_type)
        footprint = _compute(footprint, return_type)

        # The inputs do overlap the output, so there should be coverage --
        # asserting this makes the test non-vacuous (a fully blank output would
        # otherwise satisfy the check below trivially).
        assert np.any(footprint > 0)
        assert np.all((array != 0) == (footprint > 0))

    @pytest.mark.parametrize("combine_function", ["mean", "sum", "first", "last", "min", "max"])
    def test_background_matching_consistent_tiles(self, reproject_function, combine_function):
        # When the input tiles are mutually consistent (here cut from a single
        # array), background matching computes essentially zero corrections, so
        # enabling it must not change the output. This exercises background
        # matching together with every combine function -- in particular
        # first/last/min/max, which must still combine the (corrected) images
        # rather than producing a blank output.
        input_data = self._get_tiles(self._overlapping_views)

        kwargs = dict(
            shape_out=self.array.shape,
            combine_function=combine_function,
            reproject_function=reproject_function,
        )
        array_nomatch, footprint_nomatch = reproject_and_coadd(
            input_data, self.wcs, match_background=False, **kwargs
        )
        array_match, footprint_match = reproject_and_coadd(
            input_data, self.wcs, match_background=True, **kwargs
        )

        # The matched output must actually have coverage and agree with the
        # unmatched output.
        assert np.any(footprint_match > 0)
        np.testing.assert_allclose(footprint_match, footprint_nomatch, atol=ATOL)
        np.testing.assert_allclose(array_match, array_nomatch, atol=ATOL)

    def test_coadd_background_matching_with_nan(
        self, reproject_function, intermediate_memmap_nozarr
    ):
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
            intermediate_memmap=intermediate_memmap_nozarr,
        )

        # The absolute values of the two arrays will be offset since any
        # solution that reproduces the offsets between images is valid

        assert_allclose(array - np.mean(array), self.array - np.mean(self.array), atol=ATOL)

    @pytest.mark.filterwarnings("ignore:unclosed file:ResourceWarning")
    @pytest.mark.parametrize("mode", ["arrays", "filenames", "hdus", "hdulist"])
    def test_coadd_with_weights(
        self, tmpdir, reproject_function, mode, intermediate_memmap, return_type
    ):
        # Make sure that things work properly when specifying weights

        if return_type == "dask" and intermediate_memmap:
            pytest.skip("return_type='dask' does not support intermediate_memmap")

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
            return_type=return_type,
            intermediate_memmap=intermediate_memmap,
        )
        array = _compute(array, return_type)

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
            intermediate_memmap=intermediate_memmap,
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
            intermediate_memmap=intermediate_memmap,
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


@pytest.mark.filterwarnings("ignore::erfa.ErfaWarning")
@pytest.mark.parametrize("combine_function", ["mean", "sum"])
@pytest.mark.parametrize("celestial_output", [False, True])
def test_coadd_non_reprojected_dims(combine_function, celestial_output):
    # Co-add cubes whose celestial coordinates drift along the non-reprojected
    # (time) axis, treating that axis as non-reprojected. With a full cube
    # output WCS the result should match co-adding without non_reprojected_dims
    # (which is an optimization and shouldn't change the answer). With a
    # celestial-only (2D) output WCS the input WCS has more pixel dimensions
    # than the output, so computing each tile's footprint requires relating
    # only the reprojected (celestial) sub-space of the input WCS to the
    # output; the result should match co-adding each time slice independently
    # with the input WCS sliced at that time.

    n_time = 5
    shape_out = (n_time, 30, 30)
    wcs_in = _drifting_cube_wcs(drift=0.6)
    wcs_out = _drifting_cube_wcs(drift=0.0)
    if celestial_output:
        wcs_out = wcs_out.celestial

    rng = np.random.default_rng(12345)
    data1 = rng.random((n_time, 30, 30))
    data2 = rng.random((n_time, 30, 30))

    # Run with non-reprojected dims
    array, footprint = reproject_and_coadd(
        [(data1, wcs_in), (data2, wcs_in)],
        wcs_out,
        shape_out=shape_out,
        reproject_function=reproject_interp,
        combine_function=combine_function,
        non_reprojected_dims=(0,),
        parallel=True,
        block_size=(1,) + shape_out[1:],
        roundtrip_coords=False,
        intermediate_memmap=True,
    )

    if celestial_output:
        reference = np.zeros(shape_out)
        reference_footprint = np.zeros(shape_out)
        for itime in range(n_time):
            reference[itime], reference_footprint[itime] = reproject_and_coadd(
                [(data1[itime], wcs_in[itime]), (data2[itime], wcs_in[itime])],
                wcs_out,
                shape_out=shape_out[1:],
                reproject_function=reproject_interp,
                combine_function=combine_function,
                roundtrip_coords=False,
            )
    else:
        # Run without non-reprojected dims (non_reprojected_dims is an
        # optimization but shouldn't give a different answer)
        reference, reference_footprint = reproject_and_coadd(
            [(data1, wcs_in), (data2, wcs_in)],
            wcs_out,
            shape_out=shape_out,
            reproject_function=reproject_interp,
            combine_function=combine_function,
            parallel=True,
            block_size=(1,) + shape_out[1:],
            roundtrip_coords=False,
            intermediate_memmap=True,
        )

    assert_allclose(array, reference, atol=ATOL)
    assert_allclose(footprint, reference_footprint, atol=ATOL)


@pytest.mark.filterwarnings("ignore::erfa.ErfaWarning")
@pytest.mark.parametrize("block_size", [(30, 30)])
def test_coadd_non_reprojected_dims_reprojected_only_block_size(block_size):
    # A block size covering only the reprojected dimensions must give the same
    # result as the full-length equivalent; the per-cutout shrinking used to
    # prepend the wrong leading entries for such block sizes. A sub-tiled
    # (12, 12) case should be added here once block sizes smaller than the
    # output along the reprojected dimensions are supported.

    n_time = 3
    shape_out = (n_time, 30, 30)
    wcs_in = _drifting_cube_wcs(drift=0.6)
    wcs_out = _drifting_cube_wcs(drift=0.0).celestial

    rng = np.random.default_rng(12345)
    data = rng.random((n_time, 30, 30))

    reference, reference_footprint = reproject_and_coadd(
        [(data, wcs_in)],
        wcs_out,
        shape_out=shape_out,
        reproject_function=reproject_interp,
        combine_function="mean",
        non_reprojected_dims=(0,),
        parallel=True,
        block_size=(1,) + shape_out[1:],
        roundtrip_coords=False,
    )

    array, footprint = reproject_and_coadd(
        [(data, wcs_in)],
        wcs_out,
        shape_out=shape_out,
        reproject_function=reproject_interp,
        combine_function="mean",
        non_reprojected_dims=(0,),
        parallel=True,
        block_size=block_size,
        roundtrip_coords=False,
    )

    assert_allclose(array, reference, atol=ATOL)
    assert_allclose(footprint, reference_footprint, atol=ATOL)


@pytest.mark.filterwarnings("ignore::erfa.ErfaWarning")
def test_coadd_non_reprojected_dims_with_weights():
    # Weights combined with non_reprojected_dims used to raise
    # NotImplementedError because the per-image weights reprojection did not
    # pass the block size through. Uniform weights should give the same result
    # as no weights.

    n_time = 3
    shape_out = (n_time, 30, 30)
    wcs_in = _drifting_cube_wcs(drift=0.6)
    wcs_out = _drifting_cube_wcs(drift=0.0)

    rng = np.random.default_rng(12345)
    data1 = rng.random((n_time, 30, 30))
    data2 = rng.random((n_time, 30, 30))

    reference, _ = reproject_and_coadd(
        [(data1, wcs_in), (data2, wcs_in)],
        wcs_out,
        shape_out=shape_out,
        reproject_function=reproject_interp,
        combine_function="mean",
        non_reprojected_dims=(0,),
        parallel=True,
        block_size=(1,) + shape_out[1:],
        roundtrip_coords=False,
    )

    array, _ = reproject_and_coadd(
        [(data1, wcs_in), (data2, wcs_in)],
        wcs_out,
        shape_out=shape_out,
        input_weights=[np.ones(data1.shape), np.ones(data2.shape)],
        reproject_function=reproject_interp,
        combine_function="mean",
        non_reprojected_dims=(0,),
        parallel=True,
        block_size=(1,) + shape_out[1:],
        roundtrip_coords=False,
    )

    assert_allclose(array, reference, atol=ATOL)


def test_coadd_non_reprojected_dims_invalid():
    # An invalid non_reprojected_dims should raise even if no input overlaps
    # the output (in which case the same check inside reproject_function is
    # never reached).
    wcs_in = _drifting_cube_wcs(drift=0.0)
    with pytest.raises(ValueError, match="increasing sequentially from zero"):
        reproject_and_coadd(
            [(np.zeros((5, 30, 30)), wcs_in)],
            wcs_in.celestial,
            shape_out=(5, 30, 30),
            reproject_function=reproject_interp,
            combine_function="mean",
            non_reprojected_dims=(1,),
        )


@pytest.mark.parametrize("combine_function", ["mean", "first", "median"])
@pytest.mark.parametrize("block_size", [(20, 20), (80, 20)])
def test_coadd_return_type_zarr(tmp_path, combine_function, block_size):
    # The zarr return type computes the same graphs as the dask return type
    # batch by batch into a store; results must match exactly, including the
    # blank fill in regions covered by no image (which for skipped batches
    # comes from the zarr fill value rather than a computed chunk). The
    # (80, 20) block size spans the full first dimension, so the batches
    # iterate over chunks of the second dimension.

    wcs1 = WCS(naxis=2)
    wcs1.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs1.wcs.crval = [40.0, 0.0]
    wcs1.wcs.cdelt = [-0.001, 0.001]
    wcs1.wcs.crpix = [20.0, 20.0]
    wcs2 = wcs1.deepcopy()
    wcs2.wcs.crpix[0] -= 15
    wcs2.wcs.crpix[1] -= 15

    rng = np.random.default_rng(42)
    input_data = [(rng.random((40, 40)), wcs1), (rng.random((40, 40)), wcs2)]

    # Output taller than the coverage so that at least one batch contains no
    # images at all and is never computed
    shape_out = (80, 60)

    kwargs = dict(
        reproject_function=reproject_interp,
        shape_out=shape_out,
        combine_function=combine_function,
        roundtrip_coords=False,
        block_size=block_size,
        blank_pixel_value=-1,
    )

    array_dask, footprint_dask = reproject_and_coadd(input_data, wcs1, return_type="dask", **kwargs)
    array_zarr, footprint_zarr = reproject_and_coadd(
        input_data,
        wcs1,
        return_type="zarr",
        zarr_path=str(tmp_path / "coadd.zarr"),
        zarr_batch_size=1,
        **kwargs,
    )

    assert_allclose(np.asarray(array_zarr), np.asarray(array_dask))
    assert_allclose(np.asarray(footprint_zarr), np.asarray(footprint_dask))


def test_coadd_return_type_zarr_non_reprojected_dims(tmp_path):
    # Batching along a non-reprojected leading dimension (the slab case)

    n_time = 6
    shape_out = (n_time, 30, 30)
    wcs_in = _drifting_cube_wcs(drift=0.6)
    wcs_out = _drifting_cube_wcs(drift=0.0)

    rng = np.random.default_rng(12345)
    input_data = [(rng.random(shape_out), wcs_in), (rng.random(shape_out), wcs_in)]

    kwargs = dict(
        reproject_function=reproject_interp,
        shape_out=shape_out,
        non_reprojected_dims=(0,),
        roundtrip_coords=False,
        block_size=(30, 30),
    )

    array_dask, footprint_dask = reproject_and_coadd(
        input_data, wcs_out, return_type="dask", **kwargs
    )
    array_zarr, footprint_zarr = reproject_and_coadd(
        input_data,
        wcs_out,
        return_type="zarr",
        zarr_path=str(tmp_path / "coadd.zarr"),
        zarr_batch_size=2,
        **kwargs,
    )

    assert_allclose(np.asarray(array_zarr), np.asarray(array_dask))
    assert_allclose(np.asarray(footprint_zarr), np.asarray(footprint_dask))


def test_coadd_return_type_zarr_parallel(tmp_path):
    # The batch computation follows the same parallel semantics as the
    # individual reprojection functions

    data = np.random.default_rng(0).random((30, 30))
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.cdelt = [-0.001, 0.001]

    kwargs = dict(
        reproject_function=reproject_interp,
        shape_out=(30, 30),
        roundtrip_coords=False,
        block_size=(10, 10),
        return_type="zarr",
    )

    results = {}
    for parallel in [False, True, 2, "current-scheduler"]:
        array, footprint = reproject_and_coadd(
            [(data, wcs)],
            wcs,
            zarr_path=str(tmp_path / f"coadd_{parallel}.zarr"),
            parallel=parallel,
            **kwargs,
        )
        results[parallel] = np.asarray(array)

    for parallel in [True, 2, "current-scheduler"]:
        assert_allclose(results[parallel], results[False])

    with pytest.raises(ValueError, match="strictly positive"):
        reproject_and_coadd(
            [(data, wcs)],
            wcs,
            zarr_path=str(tmp_path / "coadd_invalid.zarr"),
            parallel=-1,
            **kwargs,
        )


def test_coadd_return_type_zarr_validation(tmp_path):
    data = np.ones((10, 10))
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    kwargs = dict(reproject_function=reproject_interp, shape_out=(10, 10))

    with pytest.raises(ValueError, match="zarr_path should be set"):
        reproject_and_coadd([(data, wcs)], wcs, return_type="zarr", **kwargs)

    existing = tmp_path / "existing.zarr"
    existing.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        reproject_and_coadd(
            [(data, wcs)], wcs, return_type="zarr", zarr_path=str(existing), **kwargs
        )

    with pytest.raises(ValueError, match="can only be set"):
        reproject_and_coadd(
            [(data, wcs)],
            wcs,
            return_type="dask",
            zarr_path=str(tmp_path / "new.zarr"),
            **kwargs,
        )

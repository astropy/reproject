# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest
from astropy.wcs import WCS
from numpy.testing import assert_allclose

from reproject import reproject_adaptive, reproject_interp

# Reprojection functions that support non_reprojected_dims. reproject_exact can
# be added here once it gains support.
REPROJECT_FUNCTIONS = [reproject_interp, reproject_adaptive]


@pytest.fixture(params=REPROJECT_FUNCTIONS, ids=lambda func: func.__name__)
def reproject_function(request):
    return request.param


def _spectral_cube_wcs(crval_dec, crval_freq):
    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN", "FREQ"]
    wcs.wcs.crpix = [10, 10, 1]
    wcs.wcs.crval = [40.0, crval_dec, crval_freq]
    wcs.wcs.cdelt = [-0.01, 0.01, 1e6]
    return wcs


def test_non_reprojected_dims(reproject_function):
    # Reproject a cube where the input and output WCS have the same number of
    # dimensions as the data, treating the leading (spectral) axis as a
    # non-reprojected dimension. The result should match reprojecting each
    # spectral slice independently with the corresponding 2D WCS, and in
    # particular should not be affected by the (deliberately different) spectral
    # part of the WCS.

    data = np.arange(4 * 20 * 20, dtype=float).reshape((4, 20, 20))
    wcs_in = _spectral_cube_wcs(0.0, 1e9)
    wcs_out = _spectral_cube_wcs(0.02, 1e9 + 2e6)
    shape_out = (4, 20, 20)

    reference = np.empty_like(data)
    for islice in range(data.shape[0]):
        reference[islice], _ = reproject_function(
            (data[islice], wcs_in.celestial), wcs_out.celestial, shape_out=(20, 20)
        )

    array_out, _ = reproject_function(
        (data, wcs_in),
        wcs_out,
        shape_out=shape_out,
        non_reprojected_dims=(0,),
        parallel=True,
        block_size=(20, 20),
    )

    assert_allclose(array_out, reference, equal_nan=True)


@pytest.mark.parametrize("block_size", [(1, 7, 7), (7, 7), (1, 12, 20)])
def test_non_reprojected_dims_subtiled(reproject_function, block_size):
    # A block_size smaller than the output along the reprojected (celestial)
    # dimensions should reproject each plane in sub-tiles and give exactly the
    # same result as reprojecting each full plane in one go. This is what keeps
    # the coordinate-transform memory bounded for large planes.

    data = np.arange(4 * 20 * 20, dtype=float).reshape((4, 20, 20))
    wcs_in = _spectral_cube_wcs(0.0, 1e9)
    wcs_out = _spectral_cube_wcs(0.02, 1e9 + 2e6)
    shape_out = (4, 20, 20)

    array_full, footprint_full = reproject_function(
        (data, wcs_in),
        wcs_out,
        shape_out=shape_out,
        non_reprojected_dims=(0,),
        parallel=True,
        block_size=(20, 20),
    )

    array_sub, footprint_sub = reproject_function(
        (data, wcs_in),
        wcs_out,
        shape_out=shape_out,
        non_reprojected_dims=(0,),
        parallel=True,
        block_size=block_size,
    )

    assert_allclose(array_sub, array_full, equal_nan=True)
    assert_allclose(footprint_sub, footprint_full, equal_nan=True)


@pytest.mark.parametrize("chunks", [(1, 30, 30), (1, 15, 15)])
@pytest.mark.parametrize("block_size", [(20, 20), (7, 7)])
def test_non_reprojected_dims_dask_input(reproject_function, block_size, chunks):
    # A dask-array input must match the identical numpy input, both for
    # full-plane and sub-tiled blocks. With dask_method='none', an input chunked
    # one slice at a time is materialized per slice (exactly once), while an
    # input chunked below one slice is kept lazy so streaming cores never need a
    # whole slice at once; both must give the same answer. The WCS drifts along
    # the non-reprojected axis so each slice really is reprojected with its own
    # WCS.
    import dask.array as da

    n_time = 5
    shape_out = (n_time, 30, 30)
    wcs_in = _drifting_cube_wcs(drift=0.6)
    wcs_out = _drifting_cube_wcs(drift=0.0)

    data = np.random.default_rng(0).random((n_time, 30, 30))

    reference, _ = reproject_function(
        (data, wcs_in),
        wcs_out,
        shape_out=shape_out,
        non_reprojected_dims=(0,),
        parallel=True,
        block_size=(30, 30),
    )

    array_out, _ = reproject_function(
        (da.from_array(data, chunks=chunks), wcs_in),
        wcs_out,
        shape_out=shape_out,
        non_reprojected_dims=(0,),
        parallel=True,
        block_size=block_size,
        dask_method="none",
    )

    assert_allclose(array_out, reference, equal_nan=True)


def test_non_reprojected_dims_sliced_memmap(tmp_path, reproject_function):
    # A sliced memmap view keeps the parent's unadjusted .offset, so it must not
    # be reconstructed from filename and offset inside the block tasks (which
    # would silently reproject the wrong planes); views are passed by reference
    # instead. Slicing off the leading plane keeps the view c-contiguous, which
    # is the case that used to take the reconstruction path.

    data = np.arange(5 * 20 * 20, dtype=float).reshape((5, 20, 20))
    mm = np.memmap(tmp_path / "cube.np", mode="w+", dtype=float, shape=(5, 20, 20))
    mm[:] = data
    mm.flush()

    wcs_in = _spectral_cube_wcs(0.0, 1e9)
    wcs_out = _spectral_cube_wcs(0.02, 1e9 + 2e6)
    shape_out = (4, 20, 20)

    reference, _ = reproject_function(
        (data[1:], wcs_in),
        wcs_out,
        shape_out=shape_out,
        non_reprojected_dims=(0,),
        parallel=True,
        block_size=(20, 20),
    )

    array_out, _ = reproject_function(
        (mm[1:], wcs_in),
        wcs_out,
        shape_out=shape_out,
        non_reprojected_dims=(0,),
        parallel=True,
        block_size=(20, 20),
    )

    assert_allclose(array_out, reference, equal_nan=True)


def test_non_reprojected_dims_dask_input_streams_planes(reproject_function):
    # The input is passed as a dask array with one chunk per non-reprojected
    # slice, so each input plane must be computed exactly once, including when
    # the output is sub-tiled (every tile of a plane shares that plane's chunk
    # rather than recomputing it), and the whole input must never be
    # materialized at once.
    import dask.array as da

    n_time = 5
    shape_out = (n_time, 30, 30)
    wcs_in = _drifting_cube_wcs(drift=0.6)
    wcs_out = _drifting_cube_wcs(drift=0.0)

    data = np.random.default_rng(0).random((n_time, 30, 30))

    computed_planes = []

    def record_plane(plane, block_info=None):
        if block_info:
            computed_planes.append(block_info[None]["chunk-location"][0])
        return plane

    lazy = da.from_array(data, chunks=(1, 30, 30)).map_blocks(record_plane)

    array_out, _ = reproject_function(
        (lazy, wcs_in),
        wcs_out,
        shape_out=shape_out,
        non_reprojected_dims=(0,),
        parallel=True,
        block_size=(7, 7),
        dask_method="none",
    )

    reference, _ = reproject_function(
        (data, wcs_in),
        wcs_out,
        shape_out=shape_out,
        non_reprojected_dims=(0,),
        parallel=True,
        block_size=(30, 30),
    )

    assert_allclose(array_out, reference, equal_nan=True)
    assert sorted(computed_planes) == list(range(n_time))


def test_non_reprojected_dims_invalid_order(reproject_function):
    data = np.ones((4, 20, 20))
    wcs = _spectral_cube_wcs(0.0, 1e9)
    with pytest.raises(ValueError, match="increasing sequentially from zero"):
        reproject_function((data, wcs), wcs, shape_out=(4, 20, 20), non_reprojected_dims=(1,))


def test_non_reprojected_dims_inconsistent_with_wcs(reproject_function):
    # The WCS already has fewer dimensions than the data, but the shortfall does
    # not match the number of non_reprojected_dims requested.
    data = np.ones((3, 4, 20, 20))
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    with pytest.raises(ValueError, match="does not match the number of non_reprojected_dims"):
        reproject_function(
            (data, wcs),
            wcs,
            shape_out=(3, 4, 20, 20),
            non_reprojected_dims=(0,),
            parallel=True,
            block_size=(20, 20),
        )


@pytest.mark.parametrize(
    "kwargs", [{}, {"parallel": True}, {"parallel": True, "block_size": "auto"}]
)
def test_non_reprojected_dims_unsupported_mode(reproject_function, kwargs):
    # non_reprojected_dims with a full-dimensional WCS is only supported when
    # parallelizing over the non-reprojected dimensions, which requires an
    # explicit block_size; modes without one (including block_size='auto')
    # should raise rather than silently reprojecting the non-reprojected axis.
    data = np.ones((4, 20, 20))
    wcs_in = _spectral_cube_wcs(0.0, 1e9)
    wcs_out = _spectral_cube_wcs(0.02, 1e9 + 2e6)
    with pytest.raises(NotImplementedError, match="non_reprojected_dims"):
        reproject_function(
            (data, wcs_in), wcs_out, shape_out=(4, 20, 20), non_reprojected_dims=(0,), **kwargs
        )


def _drifting_cube_wcs(drift):
    # 3D WCS over (time, y, x) where the celestial axes are coupled to the time
    # pixel axis via the PC matrix, so the celestial coordinates drift along the
    # time axis (while the time axis itself stays independent of the celestial
    # axes). A drift of zero gives celestial coordinates that are constant in
    # time.
    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN", "TIME"]
    wcs.wcs.crpix = [15, 15, 1]
    wcs.wcs.crval = [40.0, 0.0, 0.0]
    wcs.wcs.cdelt = [-0.01, 0.01, 1.0]
    wcs.wcs.pc = [[1.0, 0.0, drift], [0.0, 1.0, drift], [0.0, 0.0, 1.0]]
    return wcs


def test_non_reprojected_dims_time_varying_wcs(reproject_function):
    # Motivating use case: a cube whose celestial coordinates drift along a
    # non-reprojected (time) axis, reprojected to a cube where they do not. Each
    # time slice must be reprojected using its own (drifted) celestial WCS, which
    # should match reprojecting each slice independently with the WCS sliced at
    # that time.
    n_time = 5
    shape_out = (n_time, 30, 30)
    wcs_in = _drifting_cube_wcs(drift=0.6)
    wcs_out = _drifting_cube_wcs(drift=0.0)

    data = np.random.default_rng(0).random((n_time, 30, 30))

    array_out, _ = reproject_function(
        (data, wcs_in),
        wcs_out,
        shape_out=shape_out,
        non_reprojected_dims=(0,),
        parallel=True,
        block_size=(30, 30),
    )

    reference = np.empty_like(data)
    for itime in range(n_time):
        reference[itime], _ = reproject_function(
            (data[itime], wcs_in[itime]), wcs_out[itime], shape_out=(30, 30)
        )

    assert_allclose(array_out, reference, equal_nan=True)

    # Make sure the drift is actually exercised (otherwise the test would pass
    # trivially even if a single WCS were reused for all slices).
    assert not np.allclose(np.nan_to_num(reference[0]), np.nan_to_num(reference[-1]))


@pytest.mark.filterwarnings("ignore::erfa.ErfaWarning")
def test_non_reprojected_dims_matches_full_reproject():
    # The full N-D reproject transforms the TIME axis through world coordinates,
    # which emits an incidental ERFA "dubious year" warning for this synthetic
    # epoch; that is unrelated to what we are checking here.
    # Cross-check the non_reprojected_dims fast path against a full N-D reproject
    # (with no non_reprojected_dims), which is a completely independent code path.
    # Because the time axis maps one-to-one between the input and output WCS, the
    # two must agree. This only applies to reproject_interp, since
    # reproject_adaptive does not support a full N-D reproject of a cube with a
    # coupled WCS (it is celestial-2D only).
    n_time = 5
    shape_out = (n_time, 30, 30)
    wcs_in = _drifting_cube_wcs(drift=0.6)
    wcs_out = _drifting_cube_wcs(drift=0.0)

    data = np.random.default_rng(0).random((n_time, 30, 30))

    array_out, _ = reproject_interp(
        (data, wcs_in),
        wcs_out,
        shape_out=shape_out,
        non_reprojected_dims=(0,),
        parallel=True,
        block_size=(30, 30),
    )

    reference_full, _ = reproject_interp((data, wcs_in), wcs_out, shape_out=shape_out)

    assert_allclose(array_out, reference_full, equal_nan=True, atol=1e-8)


def test_non_reprojected_dims_all_dimensions(reproject_function):
    # Marking every dimension as non-reprojected leaves nothing to reproject and
    # should raise a clear error rather than failing obscurely further down.
    data = np.ones((20, 20))
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    with pytest.raises(ValueError, match="at least one dimension"):
        reproject_function((data, wcs), wcs, shape_out=(20, 20), non_reprojected_dims=(0, 1))

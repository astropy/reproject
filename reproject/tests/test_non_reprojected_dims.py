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


def test_non_reprojected_dims_invalid_leading_block_size(reproject_function):
    # Since each block covers a single non-reprojected slice, block_size entries
    # along the non-reprojected dimensions must be 1 or the full extent; other
    # values would be silently reinterpreted as 1 so they raise instead.
    data = np.ones((4, 20, 20))
    wcs_in = _spectral_cube_wcs(0.0, 1e9)
    wcs_out = _spectral_cube_wcs(0.02, 1e9 + 2e6)
    for block_size in [(2, 20, 20), (999, 20, 20)]:
        with pytest.raises(ValueError, match="single non-reprojected slice"):
            reproject_function(
                (data, wcs_in),
                wcs_out,
                shape_out=(4, 20, 20),
                non_reprojected_dims=(0,),
                parallel=True,
                block_size=block_size,
            )


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
    "kwargs", [{}, {"parallel": True}, {"parallel": True, "block_size": (4, 10, 10)}]
)
def test_non_reprojected_dims_unsupported_mode(reproject_function, kwargs):
    # non_reprojected_dims with a full-dimensional WCS is only supported when
    # parallelizing over the non-reprojected dimensions; other modes should
    # raise rather than silently reprojecting the non-reprojected axis.
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

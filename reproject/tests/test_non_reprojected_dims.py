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

import dask.array as da
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.ndimage import map_coordinates as scipy_map_coordinates

from reproject._array_utils import (
    dask_map_coordinates,
    map_coordinates,
    pad_dask_array_to_grid,
)


@pytest.mark.parametrize("cval", [3, np.nan])
@pytest.mark.parametrize("shape", [(3, 4), (30, 40), (3, 40)])
@pytest.mark.parametrize("order", [0, 1, 2, 3])
@pytest.mark.parametrize("dtype", (">f4", ">f8", "<f4", "<f8"))
def test_custom_map_coordinates(cval, shape, order, dtype):
    np.random.seed(1249)

    data = np.random.random(shape).astype(dtype)

    coords = np.random.uniform(-2, max(shape) + 2, (2, 100000))

    expected = scipy_map_coordinates(
        np.pad(data, 1, mode="edge"),
        coords + 1,
        order=order,
        cval=cval,
        mode="constant",
    )

    reset = np.zeros(coords.shape[1], dtype=bool)

    for i in range(coords.shape[0]):
        reset |= coords[i] < -0.5
        reset |= coords[i] > data.shape[i] - 0.5

    expected[reset] = cval

    result1 = map_coordinates(
        data,
        coords,
        order=order,
        cval=cval,
        mode="constant",
    )

    # If order >= 2, the padding we used in the reference result will give
    # subtly different results, so no point in comparing this.
    if order < 2:
        assert_allclose(result1, expected)

    result2 = dask_map_coordinates(
        data,
        coords,
        order=order,
        cval=cval,
        mode="constant",
    )

    assert_allclose(result1, result2)


@pytest.mark.parametrize(
    "bounds",
    [
        [(7, 19), (18, 27)],  # interior: padding on every side
        [(0, 12), (31, 40)],  # flush with the grid edges: no padding there
        [(0, 30), (0, 9)],  # one dimension spanning the full grid
    ],
)
def test_pad_dask_array_to_grid(bounds):
    target_shape = (30, 40)
    # Non-uniform chunking along the second dimension (remainder chunk)
    target_chunks = da.core.normalize_chunks((10, 16), shape=target_shape, dtype=float)

    shape = tuple(imax - imin for (imin, imax) in bounds)
    data = np.random.default_rng(42).random(shape).astype("<f4")
    array = da.from_array(data, chunks=(5, 4))

    padded = pad_dask_array_to_grid(array, bounds, target_shape, target_chunks)

    assert padded.shape == target_shape
    assert padded.dtype == array.dtype

    expected = np.zeros(target_shape, dtype="<f4")
    expected[tuple(slice(imin, imax) for (imin, imax) in bounds)] = data
    assert_allclose(np.asarray(padded), expected)

    # The chunk boundaries are the target grid boundaries plus at most cuts at
    # the bounds of the original array, so that the rechunk to the target
    # chunking below only has to merge chunks rather than split and recombine
    # them across the whole grid.
    for idim in range(padded.ndim):
        target_edges = set(np.cumsum(target_chunks[idim]))
        result_edges = set(np.cumsum(padded.chunks[idim]))
        assert target_edges <= result_edges
        assert result_edges <= target_edges | set(bounds[idim])

    assert padded.rechunk(target_chunks).chunks == target_chunks

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.ndimage import map_coordinates as scipy_map_coordinates

from reproject.array_utils import dask_map_coordinates, map_coordinates


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

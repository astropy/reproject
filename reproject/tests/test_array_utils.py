import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.ndimage import map_coordinates as scipy_map_coordinates

from reproject._array_utils import dask_map_coordinates, map_coordinates


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


def test_map_coordinates_clips_to_coordinate_bounding_box():
    # For non-native data (as read from FITS files), map_coordinates copies the
    # data since scipy's map_coordinates copies non-native input internally.
    # The copies should be clipped to the region that the coordinates actually
    # cover, so interpolating a small region of a large array should only
    # allocate memory proportional to that region, not to the whole array.
    import tracemalloc

    np.random.seed(1249)

    data = np.random.random((64, 128, 128)).astype(">f4")  # 4 MB

    coords = np.random.uniform(20, 30, (3, 10_000))

    tracemalloc.start()
    result = map_coordinates(data, coords, order=1, cval=np.nan, mode="constant")
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # The coordinates cover a ~10x10x10 region, so with padding the copies
    # should be well under a megabyte; without clipping the whole 4 MB array
    # would be copied.
    assert peak < 1_000_000

    expected = map_coordinates(data.astype("<f4"), coords, order=1, cval=np.nan, mode="constant")
    assert_allclose(result, expected, rtol=1e-6)

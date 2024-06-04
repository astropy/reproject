import numpy as np
from numpy.testing import assert_allclose
from scipy.ndimage import map_coordinates as scipy_map_coordinates

from reproject.array_utils import map_coordinates


def test_custom_map_coordinates():
    np.random.seed(1249)

    data = np.random.random((3, 4))

    coords = np.random.uniform(-2, 6, (2, 10000))

    expected = scipy_map_coordinates(
        np.pad(data, 1, mode="edge"),
        coords + 1,
        order=1,
        cval=np.nan,
        mode="constant",
    )

    reset = np.zeros(coords.shape[1], dtype=bool)

    for i in range(coords.shape[0]):
        reset |= coords[i] < -0.5
        reset |= coords[i] > data.shape[i] - 0.5

    expected[reset] = np.nan

    result = map_coordinates(
        data,
        coords,
        order=1,
        cval=np.nan,
        mode="constant",
    )

    assert_allclose(result, expected)

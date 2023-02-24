from itertools import product

import numpy as np
import pytest

from ..overlap import compute_overlap


def test_full_overlap():
    EPS = np.radians(1e-2)
    lon, lat = np.array([[0, EPS, EPS, 0]]), np.array([[0, 0, EPS, EPS]])
    overlap, area_ratio = compute_overlap(lon, lat, lon, lat)
    np.testing.assert_allclose(overlap, EPS**2, rtol=1e-6)
    np.testing.assert_allclose(area_ratio, 1, rtol=1e-6)


def test_partial_overlap():
    EPS = np.radians(1e-2)
    ilon = np.array([[0, EPS, EPS, 0]])
    ilat = np.array([[0, 0, EPS, EPS]])
    olon = np.array([[0.5 * EPS, 1.5 * EPS, 1.5 * EPS, 0.5 * EPS]])
    olat = np.array([[0, 0, EPS, EPS]])

    overlap, area_ratio = compute_overlap(ilon, ilat, olon, olat)
    np.testing.assert_allclose(overlap, 0.5 * EPS**2, rtol=1e-6)
    np.testing.assert_allclose(area_ratio, 1, rtol=1e-6)


@pytest.mark.parametrize(("clockwise1", "clockwise2"), product([False, True], [False, True]))
def test_overlap_direction(clockwise1, clockwise2):
    # Regression test for a bug that caused the calculation to fail if one or
    # both of the polygons were clockwise

    EPS = np.radians(1e-2)
    ilon = np.array([[0, EPS, EPS, 0]])
    ilat = np.array([[0, 0, EPS, EPS]])
    olon = np.array([[0.5 * EPS, 1.5 * EPS, 1.5 * EPS, 0.5 * EPS]])
    olat = np.array([[0, 0, EPS, EPS]])

    if clockwise1:
        ilon, ilat = ilon[:, ::-1], ilat[:, ::-1]

    if clockwise2:
        olon, olat = olon[:, ::-1], olat[:, ::-1]

    overlap, area_ratio = compute_overlap(ilon, ilat, olon, olat)
    np.testing.assert_allclose(overlap, 0.5 * EPS**2, rtol=1e-6)
    np.testing.assert_allclose(area_ratio, 1, rtol=1e-6)

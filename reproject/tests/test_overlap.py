import numpy as np
from .. import compute_overlap

def test_overlap():
    EPS = np.radians(1e-2)
    lon, lat = [0, EPS, EPS, 0], [0, 0, EPS, EPS]
    overlap, area_ratio = compute_overlap(lon, lat, lon, lat)
    np.testing.assert_allclose(overlap, EPS ** 2, rtol=1e-6)
    np.testing.assert_allclose(area_ratio, 1, rtol=1e-6)

def test_overlap2():
    EPS = np.radians(1e-2)
    ilon, ilat = [0, EPS, EPS, 0], [0, 0, EPS, EPS]
    olon, olat = [0.5 * EPS, 1.5 * EPS, 1.5 * EPS, 0.5 * EPS], [0, 0, EPS, EPS]
    
    overlap, area_ratio = compute_overlap(ilon, ilat, olon, olat)
    np.testing.assert_allclose(overlap, 0.5 * EPS ** 2, rtol=1e-6)
    np.testing.assert_allclose(area_ratio, 1, rtol=1e-6)

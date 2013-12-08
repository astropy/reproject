import numpy as np
cimport numpy as np
import cython

ctypedef np.double_t DOUBLE_T

cdef extern from "overlapArea.h":
    double computeOverlap(double * ilon, double * ilat, double * olon, double * olat,
                          int energyMode, double refArea, double * areaRatio)


# @cython.wraparound(False)
# @cython.boundscheck(False)
def _compute_overlap(np.ndarray[double, ndim=2] ilon,
                     np.ndarray[double, ndim=2] ilat,
                     np.ndarray[double, ndim=2] olon,
                     np.ndarray[double, ndim=2] olat):
    cdef int i
    cdef int n = ilon.shape[0]

    cdef np.ndarray[double, ndim = 1] overlap = np.empty(n, dtype=np.double)
    cdef np.ndarray[double, ndim = 1] area_ratio = np.empty(n, dtype=np.double)

    for i in range(n):
        overlap[i] = computeOverlap(& ilon[i, 0], & ilat[i, 0], & olon[i, 0], & olat[i, 0],
                                    0, 1, & area_ratio[i])

    return overlap, area_ratio

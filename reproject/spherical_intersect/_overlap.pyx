import numpy as np
cimport numpy as np
import cython

ctypedef np.double_t DOUBLE_T

cdef extern from "overlapArea.h":
    double computeOverlap(double * ilon, double * ilat, double * olon, double * olat,
                          int energyMode, double refArea, double * areaRatio)

cdef extern from "reproject_slice_c.h":
    void _reproject_slice_c(int startx, int endx, int starty, int endy, int nx_out, int ny_out,
        double *xp_inout, double *yp_inout, double *xw_in, double *yw_in, double *xw_out, double *yw_out,
        double *array, double *array_new, double *weights,
        int col_in, int col_out, int col_array, int col_new)

# @cython.wraparound(False)
# @cython.boundscheck(False)
def _reproject_slice_cython(int startx, int endx, int starty, int endy, int nx_out, int ny_out,
    np.ndarray[double, ndim=2, mode = "c"] xp_inout,
    np.ndarray[double, ndim=2, mode = "c"] yp_inout,
    np.ndarray[double, ndim=2, mode = "c"] xw_in,
    np.ndarray[double, ndim=2, mode = "c"] yw_in,
    np.ndarray[double, ndim=2, mode = "c"] xw_out,
    np.ndarray[double, ndim=2, mode = "c"] yw_out,
    np.ndarray[double, ndim=2, mode = "c"] array,
    shape_out):

    # Create the array_new and weights objects, plus the objects needed in the loop.
    cdef np.ndarray[double, ndim = 2, mode = "c"] array_new = np.zeros(shape_out, dtype = np.double)
    cdef np.ndarray[double, ndim = 2, mode = "c"] weights = np.zeros(shape_out, dtype = np.double)

    # We need the y size of these 2-dimensional arrays in order to access the elements correctly
    # from raw C.
    cdef int col_in = xw_in.shape[1]
    cdef int col_out = xw_out.shape[1]
    cdef int col_array = array.shape[1]
    cdef int col_new = array_new.shape[1]

    # Call the C function now.
    _reproject_slice_c(startx,endx,starty,endy,nx_out,ny_out,
        &xp_inout[0,0],&yp_inout[0,0],
        &xw_in[0,0],&yw_in[0,0],&xw_out[0,0],&yw_out[0,0],&array[0,0],
        &array_new[0,0],&weights[0,0],
        col_in,col_out,col_array,col_new)

    return array_new,weights

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

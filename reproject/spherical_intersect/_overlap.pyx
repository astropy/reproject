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
        double *array, double *ilon, double *ilat, double *olon, double * olat, double *array_new, double *weights,
        double *overlap, double *area_ratio, double *original, int col_inout, int col_array, int col_new)

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
    # Arrays used in loop iterations.
    cdef np.ndarray[double, ndim = 2, mode = "c"] ilon = np.zeros((1,4), dtype = np.double)
    cdef np.ndarray[double, ndim = 2, mode = "c"] ilat = np.zeros((1,4), dtype = np.double)
    cdef np.ndarray[double, ndim = 2, mode = "c"] olon = np.zeros((1,4), dtype = np.double)
    cdef np.ndarray[double, ndim = 2, mode = "c"] olat = np.zeros((1,4), dtype = np.double)
    cdef np.ndarray[double, ndim = 1, mode = "c"] overlap = np.zeros((ilon.shape[0]), dtype = np.double)
    cdef np.ndarray[double, ndim = 1, mode = "c"] area_ratio = np.zeros((ilon.shape[0]), dtype = np.double)
    cdef np.ndarray[double, ndim = 1, mode = "c"] original = np.zeros((ilon.shape[0]), dtype = np.double)

    # We need the y size of these 2-dimensional arrays in order to access the elements correctly
    # from raw C.
    cdef int col_inout = xw_in.shape[1]
    cdef int col_array = array.shape[1]
    cdef int col_new = array_new.shape[1]

    # Call the C function now.
    _reproject_slice_c(startx,endx,starty,endy,nx_out,ny_out,
        &xp_inout[0,0],&yp_inout[0,0],
        &xw_in[0,0],&yw_in[0,0],&xw_out[0,0],&yw_out[0,0],&array[0,0],
        &ilon[0,0],&ilat[0,0],&olon[0,0],&olat[0,0],&array_new[0,0],&weights[0,0],
        &overlap[0],&area_ratio[0],&original[0],col_inout,col_array,col_new)

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

# @cython.wraparound(False)
# @cython.boundscheck(False)
def _reproject_loop_wrapper(int nx_in,
                            int ny_in,
                            int nx_out,
                            int ny_out,
                            np.ndarray[double, ndim=2] xp_inout,
                            np.ndarray[double, ndim=2] yp_inout,
                            np.ndarray[double, ndim=2] xw_in,
                            np.ndarray[double, ndim=2] xw_out,
                            np.ndarray[double, ndim=2] yw_in,
                            np.ndarray[double, ndim=2] yw_out,
                            np.ndarray[double, ndim=2] array_new,
                            np.ndarray[double, ndim=2] weights,
                            array):

    cdef int xmin, xmax, ymin, ymax, i, j, ii, jj
    cdef np.ndarray[double, ndim=2] ilon, ilat, olon, olat
    cdef np.ndarray[double, ndim=1] overlap, original

    for i in range(nx_in):
        for j in range(ny_in):

            # For every input pixel we find the position in the output image in
            # pixel coordinates, then use the full range of overlapping output
            # pixels with the exact overlap function.

            xmin = int(min(xp_inout[j, i], xp_inout[j, i+1], xp_inout[j+1, i+1], xp_inout[j+1, i]))
            xmax = int(max(xp_inout[j, i], xp_inout[j, i+1], xp_inout[j+1, i+1], xp_inout[j+1, i]))
            ymin = int(min(yp_inout[j, i], yp_inout[j, i+1], yp_inout[j+1, i+1], yp_inout[j+1, i]))
            ymax = int(max(yp_inout[j, i], yp_inout[j, i+1], yp_inout[j+1, i+1], yp_inout[j+1, i]))

            ilon = np.array([[xw_in[j, i], xw_in[j, i+1], xw_in[j+1, i+1], xw_in[j+1, i]][::-1]])
            ilat = np.array([[yw_in[j, i], yw_in[j, i+1], yw_in[j+1, i+1], yw_in[j+1, i]][::-1]])
            ilon = np.radians(ilon)
            ilat = np.radians(ilat)

            xmin = max(0, xmin)
            xmax = min(nx_out-1, xmax)
            ymin = max(0, ymin)
            ymax = min(ny_out-1, ymax)

            for ii in range(xmin, xmax+1):
                for jj in range(ymin, ymax+1):

                    olon = np.array([[xw_out[jj, ii], xw_out[jj, ii+1], xw_out[jj+1, ii+1], xw_out[jj+1, ii]][::-1]])
                    olat = np.array([[yw_out[jj, ii], yw_out[jj, ii+1], yw_out[jj+1, ii+1], yw_out[jj+1, ii]][::-1]])
                    olon = np.radians(olon)
                    olat = np.radians(olat)

                    # Figure out the fraction of the input pixel that makes it
                    # to the output pixel at this position.

                    overlap, _ = _compute_overlap(ilon, ilat, olon, olat)
                    original, _ = _compute_overlap(ilon, ilat, ilon, ilat)
                    array_new[jj, ii] += array[j, i] * overlap / original
                    weights[jj, ii] += overlap / original

# @cython.wraparound(False)
# @cython.boundscheck(False)
def _reproject_par_func(int start,
                        int end,
                        int ny_in,
                        np.ndarray[double, ndim=2] xp_inout,
                        np.ndarray[double, ndim=2] yp_inout,
                        np.ndarray[double, ndim=2] xw_in,
                        np.ndarray[double, ndim=2] yw_in,
                        np.ndarray[double, ndim=2] xw_out,
                        np.ndarray[double, ndim=2] yw_out,
                        int nx_out,
                        int ny_out,
                        array,
                        shape_out):

    cdef np.ndarray[double, ndim=2] array_new = np.zeros(shape_out)
    cdef np.ndarray[double, ndim=2] weights = np.zeros(shape_out)
    cdef np.ndarray[double, ndim=2] ilon, ilat, olon, olat
    cdef np.ndarray[double, ndim=1] overlap, original

    cdef int i, j, xmin, xmax, ymin, ymax, ii, jj

    for i in range(start,end):
        for j in range(ny_in):

            # For every input pixel we find the position in the output image in
            # pixel coordinates, then use the full range of overlapping output
            # pixels with the exact overlap function.

            xmin = int(min(xp_inout[j, i], xp_inout[j, i+1], xp_inout[j+1, i+1], xp_inout[j+1, i]))
            xmax = int(max(xp_inout[j, i], xp_inout[j, i+1], xp_inout[j+1, i+1], xp_inout[j+1, i]))
            ymin = int(min(yp_inout[j, i], yp_inout[j, i+1], yp_inout[j+1, i+1], yp_inout[j+1, i]))
            ymax = int(max(yp_inout[j, i], yp_inout[j, i+1], yp_inout[j+1, i+1], yp_inout[j+1, i]))

            ilon = np.array([[xw_in[j, i], xw_in[j, i+1], xw_in[j+1, i+1], xw_in[j+1, i]][::-1]])
            ilat = np.array([[yw_in[j, i], yw_in[j, i+1], yw_in[j+1, i+1], yw_in[j+1, i]][::-1]])
            ilon = np.radians(np.array(ilon))
            ilat = np.radians(np.array(ilat))

            xmin = max(0, xmin)
            xmax = min(nx_out-1, xmax)
            ymin = max(0, ymin)
            ymax = min(ny_out-1, ymax)

            for ii in range(xmin, xmax+1):
                for jj in range(ymin, ymax+1):

                    olon = np.array([[xw_out[jj, ii], xw_out[jj, ii+1], xw_out[jj+1, ii+1], xw_out[jj+1, ii]][::-1]])
                    olat = np.array([[yw_out[jj, ii], yw_out[jj, ii+1], yw_out[jj+1, ii+1], yw_out[jj+1, ii]][::-1]])
                    olon = np.radians(np.array(olon))
                    olat = np.radians(np.array(olat))

                    # Figure out the fraction of the input pixel that makes it
                    # to the output pixel at this position.

                    overlap, _ = _compute_overlap(ilon, ilat, olon, olat)
                    original, _ = _compute_overlap(ilon, ilat, ilon, ilat)
                    array_new[jj, ii] += array[j, i] * overlap / original
                    weights[jj, ii] += overlap / original

    return array_new, weights

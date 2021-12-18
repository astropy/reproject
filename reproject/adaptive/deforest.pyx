#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

# Cython implementation of the image resampling method described in "On
# resampling of Solar Images", C.E. DeForest, Solar Physics 2004

# Copyright (c) 2014, Ruben De Visscher All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sin, cos, atan2, sqrt, floor, ceil, round, exp, fabs
import sys

cdef double pi = np.pi
cdef double nan = np.nan

cdef extern from "math.h":
    int isnan(double x) nogil


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void svd2x2_decompose(double[:,:] M, double[:,:] U, double[:] s, double[:,:] V) nogil:
    cdef double E = (M[0,0] + M[1,1]) / 2
    cdef double F = (M[0,0] - M[1,1]) / 2
    cdef double G = (M[1,0] + M[0,1]) / 2
    cdef double H = (M[1,0] - M[0,1]) / 2
    cdef double Q = sqrt(E*E + H*H)
    cdef double R = sqrt(F*F + G*G)
    s[0] = Q + R
    s[1] = Q - R
    cdef double a1 = atan2(G,F)
    cdef double a2 = atan2(H,E)
    cdef double theta = (a2 - a1) / 2
    cdef double phi = (a2 + a1) / 2
    U[0,0] = cos(phi)
    U[0,1] = -sin(phi)
    U[1,0] = sin(phi)
    U[1,1] = cos(phi)
    V[0,0] = cos(theta)
    V[0,1] = sin(theta)
    V[1,0] = -sin(theta)
    V[1,1] = cos(theta)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void mul2x2(double[:,:] A, double[:,:] B, double[:,:] C) nogil:
    C[0,0] = A[0,0] * B[0,0] + A[0,1] * B[1,0]
    C[0,1] = A[0,0] * B[0,1] + A[0,1] * B[1,1]
    C[1,0] = A[1,0] * B[0,0] + A[1,1] * B[1,0]
    C[1,1] = A[1,0] * B[0,1] + A[1,1] * B[1,1]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double det2x2(double[:,:] M) nogil:
    return M[0,0]*M[1,1] - M[0,1]*M[1,0]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void svd2x2_compose(double[:,:] U, double[:] s, double[:,:] V, double[:,:] M) nogil:
    cdef double tmp00, tmp01, tmp10, tmp11
    tmp00 = U[0,0] * s[0]
    tmp01 = U[0,1] * s[1]
    tmp10 = U[1,0] * s[0]
    tmp11 = U[1,1] * s[1]
    # Multiply with transpose of V
    M[0,0] = tmp00 * V[0,0] + tmp01 * V[0,1]
    M[0,1] = tmp00 * V[1,0] + tmp01 * V[1,1]
    M[1,0] = tmp10 * V[0,0] + tmp11 * V[0,1]
    M[1,1] = tmp10 * V[1,0] + tmp11 * V[1,1]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double hanning_filter(double x, double y) nogil:
    x = fabs(x)
    y = fabs(y)
    if x >= 1 or y >= 1:
        return 0
    return (cos(x * pi)+1.0) * (cos(y * pi)+1.0) / 2.0


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double gaussian_filter(double x, double y) nogil:
    return exp(-(x*x+y*y) * 1.386294)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double clip(double x, double vmin, double vmax, int cyclic, int out_of_range_nan) nogil:
    if x < vmin:
        if cyclic:
            while x < vmin:
                x += (vmax-vmin)+1
        elif out_of_range_nan:
            return nan
        else:
            return vmin
    elif x > vmax:
        if cyclic:
            while x > vmax:
                x -= (vmax-vmin)+1
        elif out_of_range_nan:
            return nan
        else:
            return vmax
    else:
        return x


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double bilinear_interpolation(double[:,:] source, double x, double y, int x_cyclic, int y_cyclic, int out_of_range_nan) nogil:

    x = clip(x, -0.5, source.shape[1] - 0.5, x_cyclic, out_of_range_nan)
    y = clip(y, -0.5, source.shape[0] - 0.5, y_cyclic, out_of_range_nan)

    if isnan(x) or isnan(y):
        return nan

    cdef int xmin = <int>floor(x)
    cdef int ymin = <int>floor(y)
    cdef int xmax = xmin + 1
    cdef int ymax = ymin + 1

    cdef double fQ11 = source[max(0, ymin), max(0, xmin)]
    cdef double fQ21 = source[max(0, ymin), min(source.shape[1] - 1, xmax)]
    cdef double fQ12 = source[min(source.shape[0] - 1, ymax), max(0, xmin)]
    cdef double fQ22 = source[min(source.shape[0] - 1, ymax), min(source.shape[1] - 1, xmax)]

    return ((fQ11 * (xmax - x) * (ymax - y)
             + fQ21 * (x - xmin) * (ymax - y)
             + fQ12 * (xmax - x) * (y - ymin)
             + fQ22 * (x - xmin) * (y - ymin))
            * ((xmax - xmin) * (ymax - ymin)))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double nearest_neighbour_interpolation(double[:,:] source, double x, double y, int x_cyclic, int y_cyclic, int out_of_range_nan) nogil:
    y = clip(round(y), 0, source.shape[0]-1, y_cyclic, out_of_range_nan)
    x = clip(round(x), 0, source.shape[1]-1, x_cyclic, out_of_range_nan)
    if isnan(y) or isnan(x):
        return nan
    return source[<int>y, <int>x]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def map_coordinates(double[:,:] source, double[:,:] target, Ci, int max_samples_width=-1,
                    int conserve_flux=False, int progress=False, int singularities_nan=False,
                    int x_cyclic=False, int y_cyclic=False, int out_of_range_nan=False,
                    int order=0, bint center_jacobian=False):
    cdef np.ndarray[np.float64_t, ndim=3] pixel_target
    cdef int delta
    if center_jacobian:
        pixel_target = np.zeros((target.shape[0], target.shape[1], 2))
        delta = 0
    else:
        # Pad by one on all four sides of the array, so we can interpolate
        # Jacobian values from both directions at all points.
        pixel_target = np.zeros((target.shape[0]+2, target.shape[1]+2, 2))
        # With this delta set, the value of pixel_target at (0,0) will really
        # be representing (-1,-1) in the output image.
        delta = -1

    cdef int yi, xi, yoff, xoff
    for yi in range(pixel_target.shape[0]):
        for xi in range(pixel_target.shape[1]):
            pixel_target[yi,xi,0] = xi + delta
            pixel_target[yi,xi,1] = yi + delta

    cdef np.ndarray[np.float64_t, ndim=3] offset_target_x
    cdef np.ndarray[np.float64_t, ndim=3] offset_target_y
    if center_jacobian:
        # Prepare arrays marking coordinates offset by a half pixel, to allow
        # for calculating centered Jacobians for each output pixel by using the
        # corresponding input coordinate at locations offset by +/- 0.5 pixels.
        offset_target_x = np.zeros((target.shape[0], target.shape[1]+1, 2))
        offset_target_y = np.zeros((target.shape[0]+1, target.shape[1], 2))
        for yi in range(target.shape[0]):
            for xi in range(target.shape[1]):
                offset_target_x[yi,xi,0] = xi - 0.5
                offset_target_x[yi,xi,1] = yi
                offset_target_y[yi,xi,0] = xi
                offset_target_y[yi,xi,1] = yi - 0.5
            offset_target_x[yi,target.shape[1],0] = target.shape[1]-1 + 0.5
            offset_target_x[yi,target.shape[1],1] = yi
        for xi in range(target.shape[1]):
            offset_target_y[target.shape[0],xi,0] = xi
            offset_target_y[target.shape[0],xi,1] = target.shape[0]-1 + 0.5

    # These source arrays store a corresponding input-image coordinate for each
    # pixel in the output image.
    cdef np.ndarray[np.float64_t, ndim=3] pixel_source = Ci(pixel_target)
    cdef np.ndarray[np.float64_t, ndim=3] offset_source_x = None
    cdef np.ndarray[np.float64_t, ndim=3] offset_source_y = None
    cdef np.ndarray[np.float64_t, ndim=3] Jx = None
    cdef np.ndarray[np.float64_t, ndim=3] Jy = None

    if center_jacobian:
        offset_source_x = Ci(offset_target_x)
        offset_source_y = Ci(offset_target_y)
    else:
        # Pre-calculate the Jacobian at each pixel location, with values
        # representing the Jacobian halfway between two grid points, and thus
        # the values of Jx at [0, 0, :] representing
        # d(input coordinate)/d(output x) at (x=-.5, y=0) in the output image,
        # and Jy at [0, 0, :] representing d(input coordinate)/d(output y) at
        # (x=0,y=-.5).
        Jx = np.empty((target.shape[0], target.shape[1] + 1, 2))
        Jy = np.empty((target.shape[0] + 1, target.shape[1], 2))
        for yi in range(target.shape[0]):
            for xi in range(target.shape[1]):
                Jx[yi, xi, 0] = pixel_source[yi+1, xi, 0] - pixel_source[yi+1, xi+1, 0]
                Jx[yi, xi, 1] = pixel_source[yi+1, xi, 1] - pixel_source[yi+1, xi+1, 1]
                Jy[yi, xi, 0] = pixel_source[yi, xi+1, 0] - pixel_source[yi+1, xi+1, 0]
                Jy[yi, xi, 1] = pixel_source[yi, xi+1, 1] - pixel_source[yi+1, xi+1, 1]
            xi = target.shape[1]
            Jx[yi, xi, 0] = pixel_source[yi+1, xi, 0] - pixel_source[yi+1, xi+1, 0]
            Jx[yi, xi, 1] = pixel_source[yi+1, xi, 1] - pixel_source[yi+1, xi+1, 1]
        yi = target.shape[0]
        for xi in range(target.shape[1]):
            Jy[yi, xi, 0] = pixel_source[yi, xi+1, 0] - pixel_source[yi+1, xi+1, 0]
            Jy[yi, xi, 1] = pixel_source[yi, xi+1, 1] - pixel_source[yi+1, xi+1, 1]

        # Now trim the padding we added earlier. Since `delta` was used above,
        # the value at (0,0) will now truly represent (0,0) and so on. After
        # this point, pixel_source is the same for both the centered and
        # uncentered Jacobian paths.
        pixel_source = pixel_source[1:-1, 1:-1]

    cdef double[:,:] Ji = np.zeros((2, 2))
    cdef double[:,:] Ji_padded = np.zeros((2, 2))
    cdef double[:,:] J = np.zeros((2, 2))
    cdef double[:,:] U = np.zeros((2, 2))
    cdef double[:] s = np.zeros((2,))
    cdef double[:] s_padded = np.zeros((2,))
    cdef double[:] si = np.zeros((2,))
    cdef double[:,:] V = np.zeros((2, 2))
    cdef int samples_width
    cdef double[:] transformed = np.zeros((2,))
    cdef double[:] current_pixel_source = np.zeros((2,))
    cdef double[:] current_offset = np.zeros((2,))
    cdef double weight_sum = 0.0
    cdef double weight
    cdef double interpolated
    cdef double[:] P1 = np.empty((2,))
    cdef double[:] P2 = np.empty((2,))
    cdef double[:] P3 = np.empty((2,))
    cdef double[:] P4 = np.empty((2,))
    cdef double top, bottom, left, right
    cdef bint has_sampled_this_row
    with nogil:
        # Iterate through each pixel in the output image.
        for yi in range(target.shape[0]):
            for xi in range(target.shape[1]):
                if center_jacobian:
                    # Compute the Jacobian for the transformation applied to
                    # this pixel, as finite differences.
                    Ji[0,0] = offset_source_x[yi, xi, 0] - offset_source_x[yi, xi+1, 0]
                    Ji[1,0] = offset_source_x[yi, xi, 1] - offset_source_x[yi, xi+1, 1]
                    Ji[0,1] = offset_source_y[yi, xi, 0] - offset_source_y[yi+1, xi, 0]
                    Ji[1,1] = offset_source_y[yi, xi, 1] - offset_source_y[yi+1, xi, 1]
                else:
                    # Compute the Jacobian for the transformation applied to
                    # this pixel, as a mean of the Jacobian a half-pixel
                    # forwards and backwards.
                    Ji[0,0] = (Jx[yi, xi, 0] + Jx[yi, xi+1, 0]) / 2
                    Ji[1,0] = (Jx[yi, xi, 1] + Jx[yi, xi+1, 1]) / 2
                    Ji[0,1] = (Jy[yi, xi, 0] + Jy[yi+1, xi, 0]) / 2
                    Ji[1,1] = (Jy[yi, xi, 1] + Jy[yi+1, xi, 1]) / 2
                if isnan(Ji[0,0]) or isnan(Ji[0,1]) or isnan(Ji[1,0]) or isnan(Ji[1,1]) or isnan(pixel_source[yi,xi,0]) or isnan(pixel_source[yi,xi,1]):
                    target[yi,xi] = nan
                    continue

                # Find and pad the singular values of the Jacobian.
                svd2x2_decompose(Ji, U, s, V)
                s_padded[0] = max(1.0, s[0])
                s_padded[1] = max(1.0, s[1])
                si[0] = 1.0/s[0]
                si[1] = 1.0/s[1]
                svd2x2_compose(V, si, U, J)
                svd2x2_compose(U, s_padded, V, Ji_padded)

                # We'll need to sample some number of input images to set this
                # output pixel. Later on, we'll compute weights to assign to
                # each input pixel with a Hanning window, and that window will
                # assign weights of zero outside some range. Right now, we'll
                # determine a search region within the input image---a bounding
                # box around those pixels that will be assigned non-zero
                # weights.
                #
                # We do that by identifying the locations in the input image of
                # the corners of a square region centered around the output
                # pixel (using the local linearization of the transformation).
                # Those transformed coordinates will set our bounding box.
                #
                # The output-plane region we're transforming is twice the width
                # of a pixel---it runs to the centers of the neighboring
                # pixels, rather than the edges of those pixels. When we use
                # the Hann window as our filter function, having that window
                # stretch to the neighboring pixel centers ensures that, at
                # every point, the sum of the overlapping Hann windows is 1,
                # and therefore that every input-image pixel is fully
                # distributed into some combination of output pixels (in the
                # limit of a Jacobian that is constant across all output
                # pixels).

                # Transform the corners of the output-plane region to the input
                # plane.
                P1[0] = - 1 * Ji_padded[0, 0] + 1 * Ji_padded[0, 1]
                P1[1] = - 1 * Ji_padded[1, 0] + 1 * Ji_padded[1, 1]
                P2[0] = + 1 * Ji_padded[0, 0] + 1 * Ji_padded[0, 1]
                P2[1] = + 1 * Ji_padded[1, 0] + 1 * Ji_padded[1, 1]
                P3[0] = - 1 * Ji_padded[0, 0] - 1 * Ji_padded[0, 1]
                P3[1] = - 1 * Ji_padded[1, 0] - 1 * Ji_padded[1, 1]
                P4[0] = + 1 * Ji_padded[0, 0] - 1 * Ji_padded[0, 1]
                P4[1] = + 1 * Ji_padded[1, 0] - 1 * Ji_padded[1, 1]

                # Find a bounding box around the transformed coordinates.
                # (Check all four points at each step, since sometimes negative
                # Jacobian values will mirror the transformed pixel.)
                top = max(P1[1], P2[1], P3[1], P4[1])
                bottom = min(P1[1], P2[1], P3[1], P4[1])
                right = max(P1[0], P2[0], P3[0], P4[0])
                left = min(P1[0], P2[0], P3[0], P4[0])

                if max_samples_width > 0 and max(right-left, top-bottom) > max_samples_width:
                    if singularities_nan:
                        target[yi,xi] = nan
                    else:
                        if order == 0:
                            target[yi,xi] = nearest_neighbour_interpolation(source, pixel_source[yi,xi,0], pixel_source[yi,xi,1], x_cyclic, y_cyclic, out_of_range_nan)
                        else:
                            target[yi,xi] = bilinear_interpolation(source, pixel_source[yi,xi,0], pixel_source[yi,xi,1], x_cyclic, y_cyclic, out_of_range_nan)
                    continue

                # Clamp to the largest offsets that remain within the source
                # image. (Going out-of-bounds in the image plane would be
                # handled correctly in the interpolation routines, but skipping
                # those pixels altogether is faster.)
                if not x_cyclic:
                    right = min(source.shape[1] - 0.5 - pixel_source[yi,xi,0], right)
                    left = max(-0.5 - pixel_source[yi,xi,0], left)
                if not y_cyclic:
                    top = min(source.shape[0] - 0.5 - pixel_source[yi,xi,1], top)
                    bottom = max(-0.5 - pixel_source[yi,xi,1], bottom)

                target[yi,xi] = 0.0
                weight_sum = 0.0

                # Iterate through that bounding box in the input image.
                for yoff in range(<int>ceil(bottom), <int>floor(top)+1):
                    current_offset[1] = yoff
                    current_pixel_source[1] = pixel_source[yi,xi,1] + yoff
                    has_sampled_this_row = False
                    for xoff in range(<int>ceil(left), <int>floor(right)+1):
                        current_offset[0] = xoff
                        current_pixel_source[0] = pixel_source[yi,xi,0] + xoff
                        # Find the fractional position of the input location
                        # within the transformed ellipse.
                        transformed[0] = J[0,0] * current_offset[0] + J[0,1] * current_offset[1]
                        transformed[1] = J[1,0] * current_offset[0] + J[1,1] * current_offset[1]

                        # Compute an averaging weight to be assigned to this
                        # input location.
                        weight = hanning_filter(transformed[0], transformed[1])
                        if weight == 0:
                            # As we move along each row in the image, we'll
                            # first be seeing input-plane pixels that don't map
                            # back into the desired output region (i.e. they
                            # fall outside the Hanning window), then we'll see
                            # pixels that do get sampled for our output-plane
                            # pixel, and then we'll see more that don't. Once
                            # we're seeing that second group, we know we've
                            # found everything of interest in the row and can
                            # end this inner loop early. (One could be smart
                            # about skipping that first group of unused pixels,
                            # but doing so is less trivial, and skipping the
                            # second group is already only a small gain.)
                            if has_sampled_this_row:
                                break
                            continue

                        # Produce an input-image value to sample. Our output
                        # pixel doesn't necessarily map to an integer
                        # coordinate in the input image, and so our input
                        # samples must be interpolated.
                        if order == 0:
                            interpolated = nearest_neighbour_interpolation(source, current_pixel_source[0], current_pixel_source[1], x_cyclic, y_cyclic, out_of_range_nan)
                        else:
                            interpolated = bilinear_interpolation(source, current_pixel_source[0], current_pixel_source[1], x_cyclic, y_cyclic, out_of_range_nan)
                        if not isnan(interpolated):
                            target[yi,xi] += weight * interpolated
                            weight_sum += weight
                target[yi,xi] /= weight_sum
                if conserve_flux:
                    target[yi,xi] *= fabs(det2x2(Ji))
            if progress:
                with gil:
                    sys.stdout.write("\r%d/%d done" % (yi+1, target.shape[0]))
                    sys.stdout.flush()
    if progress:
        sys.stdout.write("\n")

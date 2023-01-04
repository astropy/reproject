#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

# Cython implementation of the image resampling method described in "On
# resampling of Solar Images", C.E. DeForest, Solar Physics 2004

# Original version copyright (c) 2014, Ruben De Visscher. All rights reserved.
# v2 updates copyright (c) 2022, Sam Van Kooten. All rights reserved.
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

cimport cython
cimport numpy as np
from libc.math cimport atan2, ceil, cos, exp, fabs, floor, round, sin, sqrt

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
    return (cos(x * pi)+1.0) * (cos(y * pi)+1.0)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double gaussian_filter(double x, double y, double width) nogil:
    return exp(-(x*x+y*y) / (width*width) * 2)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double clip(double x, double vmin, double vmax, int cyclic,
        int out_of_range_nearest) nogil:
    if x < vmin:
        if cyclic:
            while x < vmin:
                x += (vmax-vmin)+1
        elif out_of_range_nearest:
            return vmin
        else:
            return nan
    elif x > vmax:
        if cyclic:
            while x > vmax:
                x -= (vmax-vmin)+1
        elif out_of_range_nearest:
            return vmax
        else:
            return nan
    return x


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef bint sample_array(double[:,:,:] source, double[:] dest,
        double x, double y, int x_cyclic, int y_cyclic,
        bint out_of_range_nearest) nogil:
    x = clip(x, 0, source.shape[2] - 1, x_cyclic, out_of_range_nearest)
    y = clip(y, 0, source.shape[1] - 1, y_cyclic, out_of_range_nearest)

    if isnan(x) or isnan(y):
        return False

    # Cython doesn't like a return type of (double[:], bint), so we put the
    # input data into the provided output array
    dest[:] = source[:, <int> y, <int> x]
    return True


KERNELS = {}
KERNELS['hann'] = 0
KERNELS['hanning'] = KERNELS['hann']
KERNELS['gaussian'] = 1

BOUNDARY_MODES = {}
BOUNDARY_MODES['strict'] = 1
BOUNDARY_MODES['constant'] = 2
BOUNDARY_MODES['grid-constant'] = 3
BOUNDARY_MODES['ignore'] = 4
BOUNDARY_MODES['ignore_threshold'] = 5
BOUNDARY_MODES['nearest'] = 6


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def map_coordinates(double[:,:,:] source, double[:,:,:] target, Ci, int max_samples_width=-1,
                    int conserve_flux=False, int progress=False, int singularities_nan=False,
                    int x_cyclic=False, int y_cyclic=False, int out_of_range_nan=False,
                    bint center_jacobian=False, str kernel='gaussian', double kernel_width=1.3,
                    double sample_region_width=4, str boundary_mode="strict",
                    double boundary_fill_value=0, double boundary_ignore_threshold=0.5):
    # n.b. the source and target arrays are expected to contain three
    # dimensions---the last two are the image dimensions, while the first
    # indexes multiple images with the same coordinates. The transformation is
    # computed once, and then each image is reprojected using that
    # transformation. For the single-image case, the first dimension is still
    # required and will have size 1.
    cdef int kernel_flag
    try:
        kernel_flag = KERNELS[kernel.lower()]
    except KeyError:
        raise ValueError("'kernel' must be 'Hann' or 'Gaussian'")

    cdef int boundary_flag
    try:
        boundary_flag = BOUNDARY_MODES[boundary_mode.lower()]
    except KeyError:
        raise ValueError(
                f"boundary_mode '{boundary_mode}' not recognized") from None

    cdef np.ndarray[np.float64_t, ndim=3] pixel_target
    cdef int delta
    if center_jacobian:
        pixel_target = np.zeros((target.shape[1], target.shape[2], 2))
        delta = 0
    else:
        # Pad by one on all four sides of the array, so we can interpolate
        # Jacobian values from both directions at all points.
        pixel_target = np.zeros((target.shape[1]+2, target.shape[2]+2, 2))
        # With this delta set, the value of pixel_target at (0,0) will really
        # be representing (-1,-1) in the output image.
        delta = -1

    cdef int yi, xi, y, x
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
        offset_target_x = np.zeros((target.shape[1], target.shape[2]+1, 2))
        offset_target_y = np.zeros((target.shape[1]+1, target.shape[2], 2))
        for yi in range(target.shape[1]):
            for xi in range(target.shape[2]):
                offset_target_x[yi,xi,0] = xi - 0.5
                offset_target_x[yi,xi,1] = yi
                offset_target_y[yi,xi,0] = xi
                offset_target_y[yi,xi,1] = yi - 0.5
            offset_target_x[yi,target.shape[2],0] = target.shape[2]-1 + 0.5
            offset_target_x[yi,target.shape[2],1] = yi
        for xi in range(target.shape[2]):
            offset_target_y[target.shape[1],xi,0] = xi
            offset_target_y[target.shape[1],xi,1] = target.shape[1]-1 + 0.5

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
        Jx = np.empty((target.shape[1], target.shape[2] + 1, 2))
        Jy = np.empty((target.shape[1] + 1, target.shape[2], 2))
        for yi in range(target.shape[1]):
            for xi in range(target.shape[2]):
                Jx[yi, xi, 0] = -pixel_source[yi+1, xi, 0] + pixel_source[yi+1, xi+1, 0]
                Jx[yi, xi, 1] = -pixel_source[yi+1, xi, 1] + pixel_source[yi+1, xi+1, 1]
                Jy[yi, xi, 0] = -pixel_source[yi, xi+1, 0] + pixel_source[yi+1, xi+1, 0]
                Jy[yi, xi, 1] = -pixel_source[yi, xi+1, 1] + pixel_source[yi+1, xi+1, 1]
            xi = target.shape[2]
            Jx[yi, xi, 0] = -pixel_source[yi+1, xi, 0] + pixel_source[yi+1, xi+1, 0]
            Jx[yi, xi, 1] = -pixel_source[yi+1, xi, 1] + pixel_source[yi+1, xi+1, 1]
        yi = target.shape[1]
        for xi in range(target.shape[2]):
            Jy[yi, xi, 0] = -pixel_source[yi, xi+1, 0] + pixel_source[yi+1, xi+1, 0]
            Jy[yi, xi, 1] = -pixel_source[yi, xi+1, 1] + pixel_source[yi+1, xi+1, 1]

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
    cdef double[:,:] V = np.zeros((2, 2))
    cdef int samples_width
    cdef double[:] transformed = np.zeros((2,))
    cdef double[:] current_pixel_source = np.zeros((2,))
    cdef double[:] current_offset = np.zeros((2,))
    cdef double weight_sum
    cdef double ignored_weight_sum
    cdef double weight
    cdef double[:] value = np.empty(source.shape[0])
    cdef double[:] P1 = np.empty((2,))
    cdef double[:] P2 = np.empty((2,))
    cdef double[:] P3 = np.empty((2,))
    cdef double[:] P4 = np.empty((2,))
    cdef double top, bottom, left, right
    cdef double determinant
    cdef bint has_sampled_this_row
    cdef bint is_good_sample
    with nogil:
        # Iterate through each pixel in the output image.
        for yi in range(target.shape[1]):
            for xi in range(target.shape[2]):
                if center_jacobian:
                    # Compute the Jacobian for the transformation applied to
                    # this pixel, as finite differences.
                    Ji[0,0] = -offset_source_x[yi, xi, 0] + offset_source_x[yi, xi+1, 0]
                    Ji[1,0] = -offset_source_x[yi, xi, 1] + offset_source_x[yi, xi+1, 1]
                    Ji[0,1] = -offset_source_y[yi, xi, 0] + offset_source_y[yi+1, xi, 0]
                    Ji[1,1] = -offset_source_y[yi, xi, 1] + offset_source_y[yi+1, xi, 1]
                else:
                    # Compute the Jacobian for the transformation applied to
                    # this pixel, as a mean of the Jacobian a half-pixel
                    # forwards and backwards.
                    Ji[0,0] = (Jx[yi, xi, 0] + Jx[yi, xi+1, 0]) / 2
                    Ji[1,0] = (Jx[yi, xi, 1] + Jx[yi, xi+1, 1]) / 2
                    Ji[0,1] = (Jy[yi, xi, 0] + Jy[yi+1, xi, 0]) / 2
                    Ji[1,1] = (Jy[yi, xi, 1] + Jy[yi+1, xi, 1]) / 2
                if isnan(Ji[0,0]) or isnan(Ji[0,1]) or isnan(Ji[1,0]) or isnan(Ji[1,1]) or isnan(pixel_source[yi,xi,0]) or isnan(pixel_source[yi,xi,1]):
                    target[:,yi,xi] = nan
                    continue

                # Find and pad the singular values of the Jacobian.
                svd2x2_decompose(Ji, U, s, V)
                s[0] = max(1.0, s[0])
                s[1] = max(1.0, s[1])
                svd2x2_compose(U, s, V, Ji_padded)
                # Build J, the inverse of Ji, by using 1/s and swapping the
                # order of U and V.
                s[0] = 1.0/s[0]
                s[1] = 1.0/s[1]
                svd2x2_compose(V, s, U, J)

                # We'll need to sample some number of input image pixels to set
                # this output pixel. Later on, we'll compute weights to assign
                # to each input pixel, and they will be at or near zero outside
                # some range. Right now, we'll determine a search region within
                # the input image---a bounding box around those pixels that
                # will be assigned non-zero weights.
                #
                # We do that by defining a square region in the output plane
                # centered on the output pixel, and transforming its corners to
                # the input plane (using the local linearization of the
                # transformation). Those transformed coordinates will set our
                # bounding box.
                if kernel_flag == 0:
                    # The Hann window is zero outside +/-1, so
                    # that's how far we need to go.
                    #
                    # The Hann window width is twice the width of a pixel---it
                    # runs to the centers of the neighboring pixels, rather
                    # than the edges of those pixels. This ensures that, at
                    # every point, the sum of the overlapping Hann windows is
                    # 1, and therefore that every input-image pixel is fully
                    # distributed into some combination of output pixels (in
                    # the limit of a Jacobian that is constant across all
                    # output pixels).
                    P1[0] = - 1 * Ji_padded[0, 0] + 1 * Ji_padded[0, 1]
                    P1[1] = - 1 * Ji_padded[1, 0] + 1 * Ji_padded[1, 1]
                    P2[0] = + 1 * Ji_padded[0, 0] + 1 * Ji_padded[0, 1]
                    P2[1] = + 1 * Ji_padded[1, 0] + 1 * Ji_padded[1, 1]
                    P3[0] = - 1 * Ji_padded[0, 0] - 1 * Ji_padded[0, 1]
                    P3[1] = - 1 * Ji_padded[1, 0] - 1 * Ji_padded[1, 1]
                    P4[0] = + 1 * Ji_padded[0, 0] - 1 * Ji_padded[0, 1]
                    P4[1] = + 1 * Ji_padded[1, 0] - 1 * Ji_padded[1, 1]

                    # Find a bounding box around the transformed coordinates.
                    # (Check all four points at each step, in case a negative
                    # Jacobian value is mirroring the transformed pixel.)
                    top = max(P1[1], P2[1], P3[1], P4[1])
                    bottom = min(P1[1], P2[1], P3[1], P4[1])
                    right = max(P1[0], P2[0], P3[0], P4[0])
                    left = min(P1[0], P2[0], P3[0], P4[0])
                elif kernel_flag == 1:
                    # The Gaussian window is non-zero everywhere, but it's
                    # close to zero almost everywhere. Sampling the whole input
                    # image isn't tractable, so we truncate and sample only
                    # within a certain region.
                    # n.b. `s` currently contains the reciprocal of the
                    # singular values
                    top = sample_region_width / (2 * min(s[0], s[1]))
                    bottom = -top
                    right = top
                    left = -right
                else:
                    with gil:
                        raise ValueError("Invalid kernel type")

                if max_samples_width > 0 and max(right-left, top-bottom) > max_samples_width:
                    if singularities_nan:
                        target[:,yi,xi] = nan
                    else:
                        is_good_sample = sample_array(
                                source, value, current_pixel_source[0],
                                current_pixel_source[1], x_cyclic, y_cyclic,
                                out_of_range_nearest=boundary_flag == 6)
                        if is_good_sample:
                            target[:,yi,xi] = value
                        elif boundary_flag == 2 or boundary_flag == 3:
                            target[:,yi,xi] = boundary_fill_value
                        else:
                            target[:,yi,xi] = nan
                    continue

                top += pixel_source[yi,xi,1]
                bottom += pixel_source[yi,xi,1]
                right += pixel_source[yi,xi,0]
                left += pixel_source[yi,xi,0]

                # Draw these points in to the nearest input pixel
                bottom = ceil(bottom)
                top = floor(top)
                left = ceil(left)
                right = floor(right)

                # Handle the case that the sampling region extends beyond the
                # input image boundary. For 'strict' handling, we can set the
                # output pixel to NaN right away. For 'ignore' handling, we can
                # clamp the region to exclude the out-of-bounds samples. For
                # all other boundary modes, we still need to calculate weights
                # for each out-of-bounds sample, so we do nothing here.
                if not x_cyclic:
                    if right > source.shape[2] - 1:
                        if boundary_flag == 1:
                            target[:,yi,xi] = nan
                            continue
                        if boundary_flag == 4:
                            right = source.shape[2] - 1
                    if left < 0:
                        if boundary_flag == 1:
                            target[:,yi,xi] = nan
                            continue
                        if boundary_flag == 4:
                            left = 0
                if not y_cyclic:
                    if top > source.shape[1] - 1:
                        if boundary_flag == 1:
                            target[:,yi,xi] = nan
                            continue
                        if boundary_flag == 4:
                            top = source.shape[1] - 1
                    if bottom < 0:
                        if boundary_flag == 1:
                            target[:,yi,xi] = nan
                            continue
                        if boundary_flag == 4:
                            bottom = 0

                # Check whether the sampling region falls entirely outside the
                # input image. For strict boundary handling, this is already
                # handled above by the partial case. Otherwise, we fill in an
                # appropriate value and move along. For some projections, the
                # sampling region can become very large when well outside the
                # input image, and so this detection becomes an important
                # optimization.
                if (not x_cyclic and (right < 0 or left > source.shape[2] - 1)
                        or not y_cyclic
                            and (top < 0 or bottom > source.shape[1] - 1)):
                    if boundary_flag == 3:
                        target[:,yi,xi] = boundary_fill_value
                        continue
                    if (boundary_flag == 2
                            or boundary_flag == 4
                            or boundary_flag == 5):
                        target[:,yi,xi] = nan
                        continue
                    if boundary_flag == 6:
                        # Just sample one row or column so that we get all of
                        # the nearest values. Both kernels vary independently
                        # in x and y, so sampling the full region isn't needed
                        # when the sampled values are constant in x or in y.
                        if right < left:
                            right = left
                        if top < bottom:
                            top = bottom

                target[:,yi,xi] = 0
                weight_sum = 0
                ignored_weight_sum = 0

                # Iterate through that bounding box in the input image.
                for y in range(<int> bottom, <int> top+1):
                    current_pixel_source[1] = y
                    current_offset[1] = current_pixel_source[1] - pixel_source[yi,xi,1]
                    has_sampled_this_row = False
                    for x in range(<int> left, <int> right+1):
                        current_pixel_source[0] = x
                        current_offset[0] = current_pixel_source[0] - pixel_source[yi,xi,0]
                        # Find the fractional position of the input location
                        # within the transformed ellipse.
                        transformed[0] = J[0,0] * current_offset[0] + J[0,1] * current_offset[1]
                        transformed[1] = J[1,0] * current_offset[0] + J[1,1] * current_offset[1]

                        # Compute an averaging weight to be assigned to this
                        # input location.
                        if kernel_flag == 0:
                            weight = hanning_filter(
                                    transformed[0], transformed[1])
                        elif kernel_flag == 1:
                            weight = gaussian_filter(
                                    transformed[0],
                                    transformed[1],
                                    kernel_width)
                        else:
                            with gil:
                                raise ValueError("Invalid kernel type")
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
                        has_sampled_this_row = True

                        is_good_sample = sample_array(
                                source, value, current_pixel_source[0],
                                current_pixel_source[1], x_cyclic, y_cyclic,
                                out_of_range_nearest=(boundary_flag == 6))

                        if ((boundary_flag == 2 or boundary_flag == 3)
                                and not is_good_sample):
                            value[:] = boundary_fill_value
                            is_good_sample = True

                        if is_good_sample:
                            for i in range(target.shape[0]):
                                target[i,yi,xi] += weight * value[i]
                            weight_sum += weight
                        else:
                            if boundary_flag == 5:
                                ignored_weight_sum += weight

                if (boundary_flag == 5 and
                        ignored_weight_sum / (ignored_weight_sum + weight_sum)
                            > boundary_ignore_threshold):
                    target[:,yi,xi] = nan
                else:
                    if conserve_flux:
                        determinant = fabs(det2x2(Ji))
                    for i in range(target.shape[0]):
                        target[i,yi,xi] /= weight_sum
                        if conserve_flux:
                            target[i,yi,xi] *= determinant
            if progress:
                with gil:
                    sys.stdout.write("\r%d/%d done" % (yi+1, target.shape[1]))
                    sys.stdout.flush()
    if progress:
        sys.stdout.write("\n")

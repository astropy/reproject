# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function

import numpy as np

from ..wcs_utils import convert_world_coordinates
from ..array_utils import iterate_over_celestial_slices, map_coordinates


def _reproject_celestial(array, wcs_in, wcs_out, shape_out, order=1):
    """
    Reproject data with celestial axes to a new projection using interpolation,
    assuming that the non-celestial axes match exactly and thus don't need to be
    reprojected. This is a therefore a special case where we can reproject
    all the celestial slices in the same way.

    The input and output WCS and shape have to satisfy a number of conditions:

    - The number of dimensions in each WCS should match
    - The output shape should match the dimensionality of the WCS
    - The input and output WCS should both have celestial components
    - The input and output WCS should have the same set and ordering of axis_types
    """

    # Make sure image is floating point
    array = np.asarray(array, dtype=float)

    # Check dimensionality of WCS and shape_out
    if wcs_in.wcs.naxis != wcs_out.wcs.naxis:
        raise ValueError("Number of dimensions between input and output WCS should match")
    elif len(shape_out) != wcs_out.wcs.naxis:
        raise ValueError("Length of shape_out should match number of dimensions in wcs_out")

    # Check whether celestial components are present
    if not wcs_in.has_celestial:
        raise ValueError("Input WCS does not have celestial components")
    elif not wcs_out.has_celestial:
        raise ValueError("Input WCS has celestial components but output WCS does not")

    if tuple(wcs_in.wcs.axis_types) != tuple(wcs_out.wcs.axis_types):
        raise ValueError("axis_types should match between the input and output WCS")

    if tuple(wcs_in.wcs.cunit) != tuple(wcs_out.wcs.cunit):
        raise ValueError("units should match between the input and output WCS")

    # We create an output array with the required shape, then create an array
    # that is in order of [rest, lat, lon] where rest is the flattened
    # remainder of the array. We then operate on the view, but this will change
    # the original array with the correct shape.

    array_new = np.zeros(shape_out)

    xp_in = yp_in = None

    subset = None

    # Loop over slices and interpolate
    for slice_in, slice_out in iterate_over_celestial_slices(array,
                                                             array_new,
                                                             wcs_in):

        if xp_in is None:

            # Get position of output pixel centers in input image
            xp_in, yp_in = _get_input_pixels_celestial(wcs_in.celestial,
                                                       wcs_out.celestial,
                                                       slice_out.shape)
            coordinates = np.array([yp_in.ravel(), xp_in.ravel()])

            # Now map_coordinates is actually inefficient in that if we
            # pass it a large array, it will be much slower than a small
            # array, even if we only need to reproject part of the image.
            # So here we can instead check what the bounding box of the
            # requested coordinates are. We allow for a 1-pixel padding
            # because map_coordinates needs this
            jmin, imin = np.floor(coordinates.min(axis=1)).astype(int) - 1
            jmax, imax = np.ceil(coordinates.max(axis=1)).astype(int) + 1

            ny, nx = slice_in.shape

            # Check first if we are completely outside the image. If this is
            # the case, we should just give up and return an array full of
            # NaN values
            if imin >= nx or imax < 0 or jmin >= ny or jmax < 0:
                return array_new * np.nan, array_new.astype(float)

            # Now, we check whether there is any point in defining a subset
            if imin > 0 or imax < nx - 1 or jmin > 0 or jmax < ny - 1:
                subset = (slice(max(jmin, 0), min(jmax, ny - 1)),
                          slice(max(imin, 0), min(imax, nx - 1)))
                if imin > 0:
                    coordinates[1] -= imin
                if jmin > 0:
                    coordinates[0] -= jmin


        # If possible, only consider a subset of the array for reprojection.
        # We have already adjusted the coordinates above.
        if subset is not None:
            slice_in = slice_in[subset]

        # Make sure image is floating point. We do this only now because
        # we want to avoid converting the whole input array if possible
        slice_in = np.asarray(slice_in, dtype=float)

        slice_out[:, :] = map_coordinates(slice_in,
                                          coordinates,
                                          order=order, cval=np.nan,
                                          mode='constant'
                                          ).reshape(slice_out.shape)

    return array_new, (~np.isnan(array_new)).astype(float)


def _get_input_pixels_celestial(wcs_in, wcs_out, shape_out):
    """
    Get the pixel coordinates of the pixels in an array of shape ``shape_out``
    in the input WCS.
    """

    # TODO: for now assuming that coordinates are spherical, not
    # necessarily the case. Also assuming something about the order of the
    # arguments.

    # Generate pixel coordinates of output image
    xp_out, yp_out = np.indices(shape_out, dtype=float)[::-1]

    # Convert output pixel coordinates to pixel coordinates in original image
    # (using pixel centers).
    xw_out, yw_out = wcs_out.wcs_pix2world(xp_out, yp_out, 0)

    xw_in, yw_in = convert_world_coordinates(xw_out, yw_out, wcs_out, wcs_in)

    xp_in, yp_in = wcs_in.wcs_world2pix(xw_in, yw_in, 0)

    return xp_in, yp_in

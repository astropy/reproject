# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function

import numpy as np
from astropy.wcs import WCSSUB_CELESTIAL

from ..wcs_utils import convert_world_coordinates
from ..array_utils import iterate_over_celestial_slices, pad_edge_1


def map_coordinates(image, coords, **kwargs):

    # In the built-in scipy map_coordinates, the values are defined at the
    # center of the pixels. This means that map_coordinates does not
    # correctly treat pixels that are in the outer half of the outer pixels.
    # We solve this by extending the array, updating the pixel coordinates,
    # then getting rid of values that were sampled in the range -1 to -0.5
    # and n to n - 0.5.

    from scipy.ndimage import map_coordinates as scipy_map_coordinates

    ny, nx = image.shape

    image = pad_edge_1(image)

    values = scipy_map_coordinates(image, coords + 1, **kwargs)

    reset = ((coords[0] < -0.5) | (coords[0] > ny - 0.5) |
             (coords[1] < -0.5) | (coords[1] > nx - 0.5))
    values[reset] = kwargs.get('cval', 0.)

    return values


def get_input_pixels(wcs_in, wcs_out, shape_out):
    """
    Get the pixel coordinates of the pixels in an array of shape ``shape_out``
    in the input WCS.
    """

    # TODO: for now assuming that coordinates are spherical, not
    # necessarily the case. Also assuming something about the order of the
    # arguments.

    # Generate pixel coordinates of output image
    # reversed because numpy and wcs index in opposite directions
    # z,y,x if ::1
    # x,y,z if ::-1
    pixels_out = np.indices(shape_out)[::-1]

    # Convert output pixel coordinates to pixel coordinates in original image
    # (using pixel centers).
    # x,y,z
    out_world = wcs_out.wcs_pix2world(*pixels_out, 0)

    xw_in, yw_in = convert_world_coordinates(*out_world[:2],
                                             wcs_out.celestial,
                                             wcs_in.celestial)

    xp_in, yp_in = wcs_in.celestial.wcs_world2pix(xw_in, yw_in, 0)

    input_pixels = [xp_in, yp_in,]
    if pixels_out.ndim > 2:
        input_pixels += list(pixels_out[2:])

    # x,y,z
    return input_pixels


def _reproject_celestial(array, wcs_in, wcs_out, shape_out, order=1):
    """
    Reproject data with celestial axes to a new projection using interpolation.
    """
    from scipy.ndimage import map_coordinates

    # Make sure image is floating point
    array = np.asarray(array, dtype=float)

    # For now, assume axes are independent in this routine

    # Check that WCSs are equivalent
    if (wcs_in.naxis == wcs_out.naxis and np.any(wcs_in.wcs.axis_types !=
                                                 wcs_out.wcs.axis_types)):
        raise ValueError("The input and output WCS are not equivalent")

    if shape_out[0] != array.shape[0]:
        # do full 3D interpolation
        xp_in, yp_in, zp_in = get_input_pixels(wcs_in, wcs_out,
                                               shape_out)
        coordinates = [zp_in.ravel(), yp_in.ravel(), xp_in.ravel()]
        bad_data = ~np.isfinite(array)
        array[bad_data] = 0
        array_new = map_coordinates(array, coordinates, order=order,
                                    cval=np.nan,
                                    mode='constant').reshape(shape_out)

    else:

        # We create an output array with the required shape, then create an array
        # that is in order of [rest, lat, lon] where rest is the flattened
        # remainder of the array. We then operate on the view, but this will change
        # the original array with the correct shape.

        array_new = np.zeros(shape_out)

        xp_in = yp_in = None

        # Loop over slices and interpolate
        for slice_in, slice_out in iterate_over_celestial_slices(array,
                                                                 array_new,
                                                                 wcs_in):

            if xp_in is None:  # Get position of output pixel centers in input image
                xp_in, yp_in = get_input_pixels(wcs_in.celestial,
                                                wcs_out.celestial,
                                                slice_out.shape)
                coordinates = [yp_in.ravel(), xp_in.ravel()]

            slice_out[:,:] = map_coordinates(slice_in,
                                             coordinates,
                                             order=order, cval=np.nan,
                                             mode='constant'
                                             ).reshape(slice_out.shape)

    return array_new, (~np.isnan(array_new)).astype(float)

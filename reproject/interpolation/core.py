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

    image = pad_edge_1(image)

    values = scipy_map_coordinates(image, coords + 1, **kwargs)

    reset = np.zeros(coords.shape[1], dtype=bool)

    for i in range(coords.shape[0]):
        reset |= (coords[i] < -0.5)
        reset |= (coords[i] > image.shape[i] - 0.5)

    values[reset] = kwargs.get('cval', 0.)

    return values


def _get_input_pixels_full(wcs_in, wcs_out, shape_out):
    """
    Get the pixel coordinates of the pixels in an array of shape ``shape_out``
    in the input WCS.
    """

    # Generate pixel coordinates of output image
    p_out_ax = []
    for size in shape_out:
        p_out_ax.append(np.arange(size))

    p_out = np.meshgrid(*p_out_ax, indexing='ij')

    # Convert output pixel coordinates to pixel coordinates in original image
    # (using pixel centers).
    w_out = wcs_out.wcs_pix2world(*p_out[::-1], 0)

    p_in = wcs_in.wcs_world2pix(*w_out, 0)

    return p_in[::-1]


def _get_input_pixels_celestial(wcs_in, wcs_out, shape_out):
    """
    Get the pixel coordinates of the pixels in an array of shape ``shape_out``
    in the input WCS.
    """

    # TODO: for now assuming that coordinates are spherical, not
    # necessarily the case. Also assuming something about the order of the
    # arguments.

    # Generate pixel coordinates of output image
    xp_out_ax = np.arange(shape_out[1])
    yp_out_ax = np.arange(shape_out[0])
    xp_out, yp_out = np.meshgrid(xp_out_ax, yp_out_ax)

    # Convert output pixel coordinates to pixel coordinates in original image
    # (using pixel centers).
    xw_out, yw_out = wcs_out.wcs_pix2world(xp_out, yp_out, 0)

    xw_in, yw_in = convert_world_coordinates(xw_out, yw_out, wcs_out, wcs_in)

    xp_in, yp_in = wcs_in.wcs_world2pix(xw_in, yw_in, 0)

    return xp_in, yp_in


def _reproject_celestial(array, wcs_in, wcs_out, shape_out, order=1):
    """
    Reproject data with celestial axes to a new projection using interpolation.
    """

    # Make sure image is floating point
    array = np.asarray(array, dtype=float)

    # For now, assume axes are independent in this routine

    # Check that WCSs are equivalent
    if wcs_in.naxis == wcs_out.naxis and np.any(wcs_in.wcs.axis_types != wcs_out.wcs.axis_types):
        raise ValueError("The input and output WCS are not equivalent")

    # Extract celestial part of WCS in lon/lat order
    wcs_in_celestial = wcs_in.sub([WCSSUB_CELESTIAL])
    wcs_out_celestial = wcs_out.sub([WCSSUB_CELESTIAL])

    # We create an output array with the required shape, then create an array
    # that is in order of [rest, lat, lon] where rest is the flattened
    # remainder of the array. We then operate on the view, but this will change
    # the original array with the correct shape.

    array_new = np.zeros(shape_out)

    coordinates = None

    # Loop over slices and interpolate
    for slice_in, slice_out in iterate_over_celestial_slices(array, array_new, wcs_in):

        if coordinates is None:  # Get position of output pixel centers in input image
            xp_in, yp_in = _get_input_pixels_celestial(wcs_in_celestial, wcs_out_celestial, slice_out.shape)
            coordinates = np.array([yp_in.ravel(), xp_in.ravel()])

        slice_out[:, :] = map_coordinates(slice_in,
                                          coordinates,
                                          order=order, cval=np.nan,
                                          mode='constant'
                                          ).reshape(slice_out.shape)

    return array_new, (~np.isnan(array_new)).astype(float)


def _reproject_full(array, wcs_in, wcs_out, shape_out, order=1):
    """
    Reproject n-dimensional data to a new projection using interpolation.
    """

    # Make sure image is floating point
    array = np.asarray(array, dtype=float)

    # Check that WCSs are equivalent
    if wcs_in.naxis == wcs_out.naxis and np.any(wcs_in.wcs.axis_types != wcs_out.wcs.axis_types):
        raise ValueError("The input and output WCS are not equivalent")


    # We create an output array with the required shape, then create an array
    # that is in order of [rest, lat, lon] where rest is the flattened
    # remainder of the array. We then operate on the view, but this will change
    # the original array with the correct shape.

    array_new = np.zeros(shape_out)

    p_in = _get_input_pixels_full(wcs_in, wcs_out, shape_out)

    from astropy.io import fits
    fits.writeto('p0.fits', p_in[0], clobber=True)
    fits.writeto('p1.fits', p_in[1], clobber=True)
    fits.writeto('p2.fits', p_in[2], clobber=True)

    coordinates = np.array([p.ravel() for p in p_in])

    array_new = map_coordinates(array,
                                coordinates,
                                order=order, cval=np.nan,
                                mode='constant'
                                ).reshape(shape_out)

    return array_new, (~np.isnan(array_new)).astype(float)

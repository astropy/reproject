# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function

import numpy as np
from astropy import wcs

from distutils.version import LooseVersion
NP_LT_17 = LooseVersion(np.__version__) < LooseVersion('1.7')

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
    in the input WCS, for full n-dimensional WCSes.
    """

    # Check dimensionality of WCS and shape_out
    if wcs_in.wcs.naxis != wcs_out.wcs.naxis:
        raise ValueError("Number of dimensions between input and output WCS should match")
    elif len(shape_out) != wcs_out.wcs.naxis:
        raise ValueError("Length of shape_out should match number of dimensions in wcs_out")

    # Check whether celestial components are present
    if wcs_in.has_celestial and wcs_out.has_celestial:
        has_celestial = True
    elif wcs_in.has_celestial:
        raise ValueError("Input WCS has celestial components but output WCS does not")
    elif wcs_out.has_celestial:
        raise ValueError("Output WCS has celestial components but input WCS does not")
    else:
        has_celestial = False

    # Check whether a spectral component is present, and if so, check that
    # the CTYPEs match.
    if wcs_in.wcs.spec >= 0 and wcs_out.wcs.spec >= 0:
        if wcs_in.wcs.ctype[wcs_in.wcs.spec] != wcs_out.wcs.ctype[wcs_out.wcs.spec]:
            raise ValueError("The input ({0}) and output ({1}) spectral "
                             "coordinate types are not equivalent."
                             .format(wcs_in.wcs.ctype[wcs_in.wcs.spec],
                                     wcs_out.wcs.ctype[wcs_out.wcs.spec]))
    elif wcs_in.wcs.spec >= 0:
        raise ValueError("Input WCS has a spectral component but output WCS does not")
    elif wcs_out.wcs.spec >= 0:
        raise ValueError("Output WCS has a spectral component but input WCS does not")

    # We need to make sure that either the axis types match exactly, or that
    # they are shuffled but otherwise they are unique and there is a one-to-one
    # mapping from the input to the output WCS.
    if tuple(wcs_in.wcs.axis_types) == tuple(wcs_out.wcs.axis_types):
        needs_reorder = False
    else:
        if sorted(wcs_in.wcs.axis_types) == sorted(wcs_in.wcs.axis_types):
            if len(set(wcs_in.wcs.axis_types)) < wcs_in.wcs.naxis or \
               len(set(wcs_out.wcs.axis_types)) < wcs_out.wcs.naxis:
                raise ValueError("axis_types contains non-unique elements, and "
                                 "input order does not match output order")
            else:
                needs_reorder = True
        else:
            raise ValueError("axis_types do not map from input WCS to output WCS")

    # Determine mapping from output to input WCS
    if needs_reorder:
        axis_types_in = tuple(wcs_out.wcs.axis_types)
        axis_types_out = tuple(wcs_out.wcs.axis_types)
        indices_out = [axis_types_out.index(axis_type) for axis_type in axis_types_in]
    else:
        indices_out = list(range(wcs_out.wcs.naxis))

    # Check that the units match
    for index_in, index_out in enumerate(indices_out):
        unit_in = wcs_in.wcs.cunit[index_in]
        unit_out = wcs_out.wcs.cunit[index_out]
        if unit_in != unit_out:
            raise ValueError("Units differ between input ({0}) and output "
                             "({1}) WCS".format(unit_in, unit_out))

    # Generate pixel coordinates of output image. This is reversed because
    # numpy and wcs index in opposite directions.
    pixel_out = np.indices(shape_out, dtype=float)[::-1]

    # Convert output pixel coordinates to pixel coordinates in original image
    # (using pixel centers).
    world_out = wcs_out.wcs_pix2world(*(tuple(pixel_out) + (0,)))

    if needs_reorder:

        # We start off by creating an empty array of input world coordinates, and
        # we then populate it index by index
        world_in = np.zeros_like(world_out)
        axis_types_in = list(wcs_out.wcs.axis_types)
        axis_types_out = list(wcs_out.wcs.axis_types)
        for index_in, axis_type in enumerate(axis_types_in):
            index_out = axis_types_out.index(axis_type)
            world_in[index_in] = world_out[index_out]

    else:

        world_in = world_out

    if has_celestial:

        # Now we extract the longitude and latitude from the world_out array, and
        # convert these, before converting back to pixel coordinates.
        lon_out, lat_out = world_in[wcs_in.wcs.lng], world_in[wcs_in.wcs.lat]

        # We convert these coordinates between frames
        lon_in, lat_in = convert_world_coordinates(lon_out, lat_out, wcs_out, wcs_in)

        world_in[wcs_in.wcs.lng] = lon_in
        world_in[wcs_in.wcs.lat] = lat_in

    pixel_in = wcs_in.wcs_world2pix(*(tuple(world_in) + (0,)))

    return pixel_in


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


def _reproject_celestial(array, wcs_in, wcs_out, shape_out, order=1):
    """
    Reproject data with celestial axes to a new projection using interpolation.
    """

    # Make sure image is floating point
    array = np.asarray(array, dtype=float)

    # For now, assume axes are independent in this routine

    # Check that WCSs are equivalent
    if ((wcs_in.naxis != wcs_out.naxis or
         (list(wcs_in.wcs.axis_types) != list(wcs_out.wcs.axis_types)) or
         (list(wcs_in.wcs.cunit) != list(wcs_out.wcs.cunit)))):
        raise ValueError("The input and output WCS are not equivalent")

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
            xp_in, yp_in = _get_input_pixels_celestial(wcs_in.celestial,
                                                       wcs_out.celestial,
                                                       slice_out.shape)
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

    xp_in, yp_in, zp_in = _get_input_pixels_full(wcs_in, wcs_out, shape_out)

    coordinates = np.array([p.ravel() for p in (zp_in, yp_in, xp_in)])

    array_new = map_coordinates(array,
                                coordinates,
                                order=order, cval=np.nan,
                                mode='constant'
                                ).reshape(shape_out)

    return array_new, (~np.isnan(array_new)).astype(float)

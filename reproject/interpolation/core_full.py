# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function

import numpy as np

from ..wcs_utils import convert_world_coordinates
from ..array_utils import map_coordinates


def _reproject_full(array, wcs_in, wcs_out, shape_out, order=1):
    """
    Reproject n-dimensional data to a new projection using interpolation.

    The input and output WCS and shape have to satisfy a number of conditions:

    - The number of dimensions in each WCS should match
    - The output shape should match the dimensionality of the WCS
    - The input and output WCS should have the same set of axis_types, although
      the order can be different as long as the axis_types are unique.
    """

    # Make sure image is floating point
    array = np.asarray(array, dtype=float)

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
        axis_types_in = tuple(wcs_in.wcs.axis_types)
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

    # Reshape array so that it has dimensions (npix, ndim)
    # pixel_out = pixel_out.transpose().reshape((-1, wcs_out.wcs.naxis))
    pixel_out = pixel_out.reshape((wcs_out.wcs.naxis, -1)).transpose()

    # Convert output pixel coordinates to pixel coordinates in original image
    # (using pixel centers).
    world_out = wcs_out.wcs_pix2world(pixel_out, 0)

    if needs_reorder:

        # We start off by creating an empty array of input world coordinates, and
        # we then populate it index by index
        world_in = np.zeros_like(world_out)
        axis_types_in = list(wcs_in.wcs.axis_types)
        axis_types_out = list(wcs_out.wcs.axis_types)
        for index_in, axis_type in enumerate(axis_types_in):
            index_out = axis_types_out.index(axis_type)
            world_in[:, index_in] = world_out[:, index_out]

    else:

        world_in = world_out

    if has_celestial:

        # Now we extract the longitude and latitude from the world_out array, and
        # convert these, before converting back to pixel coordinates.
        lon_out, lat_out = world_out[:, wcs_out.wcs.lng], world_out[:, wcs_out.wcs.lat]

        # We convert these coordinates between frames
        lon_in, lat_in = convert_world_coordinates(lon_out, lat_out, wcs_out, wcs_in)

        world_in[:, wcs_in.wcs.lng] = lon_in
        world_in[:, wcs_in.wcs.lat] = lat_in

    pixel_in = wcs_in.wcs_world2pix(world_in, 0)

    coordinates = pixel_in.transpose()[::-1]

    array_new = map_coordinates(array,
                                coordinates,
                                order=order, cval=np.nan,
                                mode='constant'
                                ).reshape(shape_out)

    return array_new, (~np.isnan(array_new)).astype(float)

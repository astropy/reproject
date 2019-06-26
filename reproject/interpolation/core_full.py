# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function

import numpy as np

from ..array_utils import map_coordinates


def _reproject_full(array, wcs_in, wcs_out, shape_out, order=1, array_out=None,
                    return_footprint=True):
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
    if wcs_in.low_level_wcs.pixel_n_dim != wcs_out.low_level_wcs.pixel_n_dim:
        raise ValueError("Number of dimensions between input and output WCS should match")
    elif len(shape_out) != wcs_out.low_level_wcs.pixel_n_dim:
        raise ValueError("Length of shape_out should match number of dimensions in wcs_out")

    # shape_out must be exact a tuple type
    shape_out = tuple(shape_out)

    # TODO: need to perhaps re-instate a check on spectral coordinate types, or
    # better, implement support for converting spectral coordinates in astropy.

    # Generate pixel coordinates of output image. This is reversed because
    # numpy and wcs index in opposite directions.
    pixel_out = [p.ravel() for p in np.indices(shape_out, dtype=float)]

    # Convert output pixel coordinates to pixel coordinates in original image
    # (using pixel centers).
    world_in = wcs_out.array_index_to_world(*pixel_out)

    # TODO: it would be good to avoid this if statement if possible. Also need
    # to do [;:-1] rather than use array_index since the latter rounds to an int.
    if isinstance(world_in, (list, tuple)):
        pixel_in = wcs_in.world_to_pixel(*world_in)[::-1]
    else:
        pixel_in = wcs_in.world_to_pixel(world_in)[::-1]

    pixel_in = np.array(pixel_in)

    if array_out is not None:
        if array_out.shape != tuple(shape_out):
            raise ValueError("Array sizes don't match.  Output array shape "
                             "should be {0}".format(str(tuple(shape_out))))
        elif array_out.dtype != array.dtype:
            raise ValueError("An output array of a different type than the "
                             "input array was specified, which will create an "
                             "undesired duplicate copy of the input array "
                             "in memory.")
        else:
            array_out.shape = (array_out.size,)
    else:
        array_out = np.empty(shape_out).ravel()

    map_coordinates(array, pixel_in, order=order, cval=np.nan,
                    mode='constant', output=array_out,).reshape(shape_out)

    array_out.shape = shape_out

    if return_footprint:
        return array_out, (~np.isnan(array_out)).astype(float)
    else:
        return array_out

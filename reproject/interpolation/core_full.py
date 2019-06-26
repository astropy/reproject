# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function

import numpy as np

from astropy.wcs import WCS

from ..array_utils import map_coordinates
from ..wcs_utils import efficient_pixel_to_pixel


def _reproject_full(array, wcs_in, wcs_out, shape_out, order=1, array_out=None,
                    return_footprint=True):
    """
    Reproject n-dimensional data to a new projection using interpolation.

    The input and output WCS and shape have to satisfy a number of conditions:

    - The number of dimensions in each WCS should match
    - The output shape should match the dimensionality of the WCS
    - The input and output WCS should have matching physical types, although
      the order can be different as long as the physical types are unique.
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

    if isinstance(wcs_in, WCS) and isinstance(wcs_out, WCS):

        if wcs_in.has_celestial and not wcs_out.has_celestial:
            raise ValueError("Input WCS has celestial components but output WCS does not")
        elif wcs_out.has_celestial and not wcs_in.has_celestial:
            raise ValueError("Output WCS has celestial components but input WCS does not")

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

    pixel_out = [p.ravel() for p in np.indices(shape_out, dtype=float)]
    pixel_in = efficient_pixel_to_pixel(wcs_out, wcs_in, *pixel_out[::-1])[::-1]
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

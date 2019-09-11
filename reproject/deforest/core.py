# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

from .deforest import map_coordinates
from ..wcs_utils import efficient_pixel_to_pixel, has_celestial


__all__ = ['_reproject_deforest_2d']


class CoordinateTransformer(object):

    def __init__(self, wcs_in, wcs_out):
        self.wcs_in = wcs_in
        self.wcs_out = wcs_out

    def __call__(self, input_pixel):
        xp_in, yp_in = input_pixel[:, :, 0], input_pixel[:, :, 1]
        xp_out, yp_out = efficient_pixel_to_pixel(self.wcs_out, self.wcs_in, xp_in, yp_in)
        output_pixel = np.array([xp_out, yp_out]).transpose().swapaxes(0, 1)
        return output_pixel


def _reproject_deforest_2d(array, wcs_in, wcs_out, shape_out):
    """
    Reproject celestial slices from an n-d array from one WCS to another
    using the DeForest (2003) algorithm, and assuming all other dimensions
    are independent.

    Parameters
    ----------
    array : `~numpy.ndarray`
        The array to reproject
    wcs_in : `~astropy.wcs.WCS`
        The input WCS
    wcs_out : `~astropy.wcs.WCS`
        The output WCS
    shape_out : tuple
        The shape of the output array

    Returns
    -------
    array_new : `~numpy.ndarray`
        The reprojected array
    footprint : `~numpy.ndarray`
        Footprint of the input array in the output array. Values of 0 indicate
        no coverage or valid values in the input image, while values of 1
        indicate valid values.
    """

    # Make sure image is floating point
    array_in = np.asarray(array, dtype=float)

    # Check dimensionality of WCS and shape_out
    if wcs_in.low_level_wcs.pixel_n_dim != wcs_out.low_level_wcs.pixel_n_dim:
        raise ValueError("Number of dimensions between input and output WCS should match")
    elif len(shape_out) != wcs_out.low_level_wcs.pixel_n_dim:
        raise ValueError("Length of shape_out should match number of dimensions in wcs_out")

    # Create output array
    array_out = np.zeros(shape_out)

    transformer = CoordinateTransformer(wcs_in, wcs_out)
    map_coordinates(array_in, array_out, transformer, out_of_range_nan=True)

    return array_out, (~np.isnan(array_out)).astype(float)

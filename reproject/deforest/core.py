# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from astropy.wcs import WCSSUB_CELESTIAL
from ..wcs_utils import wcs_to_celestial_frame, convert_world_coordinates
from ..array_utils import iterate_over_celestial_slices

from astropy.io import fits

__all__ = ['reproject_celestial']


class CoordinateTransformer(object):

    def __init__(self, wcs_in, wcs_out):
        self.wcs_in = wcs_in
        self.wcs_out = wcs_out

    def __call__(self, input_pixel):
        xp_in, yp_in = input_pixel[:,:,0], input_pixel[:,:,1]
        xw_in, yw_in = self.wcs_out.wcs_pix2world(xp_in, yp_in, 0)
        xw_out, yw_out = convert_world_coordinates(xw_in, yw_in, self.wcs_in, self.wcs_out)
        xp_out, yp_out = self.wcs_in.wcs_world2pix(xw_out, yw_out, 0)
        output_pixel = np.array([xp_out, yp_out]).transpose().swapaxes(0,1)
        return output_pixel


def reproject_celestial(array, wcs_in, wcs_out, shape_out):
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

    from .deforest import map_coordinates

    transformer = CoordinateTransformer(wcs_out, wcs_in)

    # Loop over slices and interpolate
    for slice_in, slice_out in iterate_over_celestial_slices(array, array_new, wcs_in):

        map_coordinates(slice_in.astype(float),
                        slice_out,
                        transformer)


    return array_new, (~np.isnan(array_new)).astype(float)

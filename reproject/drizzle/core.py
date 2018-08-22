# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function

import numpy as np

from drizzle.drizzle import drizzle

from ..wcs_utils import convert_world_coordinates

def _reproject_drizzle(array, wcs_in, wcs_out, shape_out,
                       scale="exptime", pixel_fraction=1.0, kernel="square",
                       xmin=0, xmax=0, ymin=0, ymax=0, exposure_time=1.0, units="cps"):

    # Check dimensionality of WCS and shape_out
    if wcs_in.wcs.naxis != wcs_out.wcs.naxis:
        raise ValueError("Number of dimensions between input and output WCS should match")
    elif len(shape_out) != wcs_out.wcs.naxis:
        raise ValueError("Length of shape_out should match number of dimensions in wcs_out")

    # drizzle currently obtains the output shape (shape_out) from astropy.wcs.WCS._naxis1 & _naxis2
    # This section needs to be re-worked once drizzle is corrected to except a shape param.
    wcs_out_internal = wcs_out
    if ((wcs_out_internal._naxis1 == None or wcs_out_internal._naxis1 < 1) or
        (wcs_out_internal._naxis2 == None or wcs_out_internal._naxis2 < 1)):
        # This is looks illegal (temp workaround only - if even that)
        wcs_out_internal = wcs_out_internal.copy()
        wcs_out_internal._naxis1 = shape_out[0]
        wcs_out_internal._naxis2 = shape_out[1]

    if shape_out[0] != wcs_out_internal._naxis1 or shape_out[1] != wcs_out_internal._naxis2:
        raise ValueError("shape_out should match shape of wcs_out._naxis")

    # Find any NaN values to help later with determining a footprint.
    # NB: np.isnan(array_internal.sum()) is faster than np.isnan(array_internal).any()
    initially_nan = np.isnan(array) if np.isnan(array.sum()) else False

    # The design pattern of the reproject funcs is to reproject each image individually and then
    # combine them together as a seperate operation. This is only possible with drizzle if the
    # weighting is 1 (``None`` for the drizzle API).
    # Use fill_value=np.nan to mark footprint
    drizzled_image = drizzle(array, wcs_in, wcs_out_internal,
                             scale=scale, pixel_fraction=pixel_fraction,
                             kernel=kernel, fill_value=np.nan, weight=None,
                             xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                             exposure_time=exposure_time, units=units)

    # If a value was not NaN but now is, it was added by drizzle indicating that
    # the input image DOES NOT overlap the output - this is therefore the inverse footprint.
    footprint = np.astype(~(np.isnan(drizzled_image) and not initially_nan), dtype=float)

    return drizzled_image, footprint

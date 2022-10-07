# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
WCS-related utilities
"""

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_pixel

__all__ = ["has_celestial", "pixel_to_pixel_with_roundtrip"]


def has_celestial(wcs):
    """
    Returns `True` if there are celestial coordinates in the WCS.
    """
    if isinstance(wcs, WCS):
        return wcs.has_celestial
    else:
        for world_axis_class in wcs.low_level_wcs.world_axis_object_classes.values():
            if issubclass(world_axis_class[0], SkyCoord):
                return True
        return False


def pixel_to_pixel_with_roundtrip(wcs1, wcs2, *inputs):

    outputs = pixel_to_pixel(wcs1, wcs2, *inputs)

    # Now convert back to check that coordinates round-trip, if not then set to NaN
    inputs_check = pixel_to_pixel(wcs2, wcs1, *outputs)
    reset = np.zeros(inputs_check[0].shape, dtype=bool)
    for ipix in range(len(inputs_check)):
        reset |= np.abs(inputs_check[ipix] - inputs[ipix]) > 1
    if np.any(reset):
        for ipix in range(len(inputs_check)):
            outputs[ipix] = outputs[ipix].copy()
            outputs[ipix][reset] = np.nan

    return outputs

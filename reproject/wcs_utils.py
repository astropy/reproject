# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
WCS-related utilities
"""

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_pixel, proj_plane_pixel_scales

__all__ = ["has_celestial", "pixel_to_pixel_with_roundtrip", "pixel_scale"]


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


def pixel_scale(wcs, shape):
    """
    Given a WCS and an image shape, determine the pixel scale.

    If the WCS is an :class:`~astropy.wcs.WCS` instance, the pixel scale is
    the one at the reference coordinate, otherwise it is the scale at the center
    of the image. If the pixels are not square, the smallest pixel scale is
    returned.
    """

    if isinstance(wcs, WCS):
        scales = [
            abs(s) * u for (s, u) in zip(proj_plane_pixel_scales(wcs), wcs.wcs.cunit, strict=False)
        ]
    else:
        xp, yp = (shape[1] - 1) / 2, (shape[0] - 1) / 2
        xs = np.array([xp, xp, xp + 1])
        ys = np.array([yp, yp + 1, yp])
        cs = wcs.pixel_to_world(xs, ys)
        scales = (
            abs(cs[0].separation(cs[2])),
            abs(cs[0].separation(cs[1])),
        )

    return min(*scales)

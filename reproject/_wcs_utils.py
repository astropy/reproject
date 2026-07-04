# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
WCS-related utilities
"""

import numpy as np
from astropy.coordinates import SkyCoord, SpectralCoord
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_pixel, proj_plane_pixel_scales

__all__ = ["has_celestial", "pixel_to_pixel_chunked", "pixel_scale"]


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


def has_spectral(wcs):
    """
    Returns `True` if there are spectral coordinates in the WCS.
    """
    if isinstance(wcs, WCS):
        return wcs.has_spectral
    else:
        for world_axis_class in wcs.low_level_wcs.world_axis_object_classes.values():
            if issubclass(world_axis_class[0], SpectralCoord):
                return True
        return False


def pixel_to_pixel_chunked(wcs1, wcs2, *inputs, roundtrip=False, output=None, chunk_size=200_000):
    """
    Transform pixel coordinates from ``wcs1`` to ``wcs2`` in chunks of at most
    ``chunk_size`` elements, which bounds the peak memory usage due to the
    temporary arrays allocated inside the coordinate transformations (and is
    typically also faster than a single full-size transform due to better CPU
    cache usage).

    If ``roundtrip`` is set, each chunk is also transformed back to ``wcs1``
    and coordinates that do not round-trip to within a pixel are set to NaN.

    ``output`` can be a list of arrays (one per pixel dimension of ``wcs2``,
    each with the same number of elements as the inputs) into which the
    results are written. Inputs and outputs are accessed with flat slices and
    matched element-wise in C order, so they do not need to share a shape and
    broadcast input views are never expanded into full-size arrays.
    """
    if np.isscalar(inputs[0]):
        inputs = tuple(np.array(inp) for inp in inputs)

    if output is None:
        output = [np.empty(inputs[0].shape) for _ in range(wcs2.pixel_n_dim)]

    for start in range(0, inputs[0].size, chunk_size):
        chunk = slice(start, start + chunk_size)
        chunk_inputs = [inp.flat[chunk] for inp in inputs]
        results = pixel_to_pixel(wcs1, wcs2, *chunk_inputs)
        if wcs2.pixel_n_dim == 1:
            results = [results]
        if roundtrip:
            # Convert back to check that coordinates round-trip, if not then set to NaN
            inputs_check = pixel_to_pixel(wcs2, wcs1, *results)
            if wcs1.pixel_n_dim == 1:
                inputs_check = [inputs_check]
            reset = np.zeros(inputs_check[0].shape, dtype=bool)
            for ipix in range(len(inputs_check)):
                reset |= np.abs(inputs_check[ipix] - chunk_inputs[ipix]) > 1
            if reset.any():
                results = [np.where(reset, np.nan, result) for result in results]
        for ipix in range(len(output)):
            output[ipix].flat[chunk] = results[ipix]

    return output


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
            abs(s) * u
            for (s, u) in zip(proj_plane_pixel_scales(wcs.celestial), wcs.wcs.cunit, strict=False)
        ]
    else:
        # TODO: fix the following for 3D APE-14 WCS
        xp, yp = (shape[1] - 1) / 2, (shape[0] - 1) / 2
        xs = np.array([xp, xp, xp + 1])
        ys = np.array([yp, yp + 1, yp])
        cs = wcs.pixel_to_world(xs, ys)
        scales = (
            abs(cs[0].separation(cs[2])),
            abs(cs[0].separation(cs[1])),
        )

    return min(*scales)

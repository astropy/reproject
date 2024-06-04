# Licensed under a 3-clause BSD style license - see LICENSE.rst

import warnings

import numpy as np
from astropy import units as u
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_area

from ._overlap import _reproject_slice_cython


def _reproject_celestial(
    array,
    wcs_in,
    wcs_out,
    shape_out,
    array_out=None,
    output_footprint=None,
    return_footprint=True,
):
    if array_out is None:
        array_out = np.empty(shape_out)

    if output_footprint is None:
        output_footprint = np.empty(shape_out)

    # There are currently precision issues below certain resolutions, so we
    # emit a warning if this is the case. For more details, see:
    # https://github.com/astropy/reproject/issues/199
    area_threshold = (0.05 / 3600) ** 2
    if (isinstance(wcs_in, WCS) and proj_plane_pixel_area(wcs_in) < area_threshold) or (
        isinstance(wcs_out, WCS) and proj_plane_pixel_area(wcs_out) < area_threshold
    ):
        warnings.warn(
            "The reproject_exact function currently has precision "
            "issues with images that have resolutions below ~0.05 "
            "arcsec, so the results may not be accurate.",
            UserWarning,
            stacklevel=2,
        )

    # Convert input array to float values. If this comes from a FITS, it might have
    # float32 as value type and that can break things in Cython
    array = np.asarray(array, dtype=float)
    shape_out = tuple(shape_out)

    if wcs_in.pixel_n_dim != 2:
        # TODO: make this work for n-dimensional arrays
        raise NotImplementedError("Only 2-dimensional arrays can be reprojected at this time")
    elif len(shape_out) < wcs_out.low_level_wcs.pixel_n_dim:
        raise ValueError("Too few dimensions in shape_out")
    elif len(array.shape) < wcs_in.low_level_wcs.pixel_n_dim:
        raise ValueError("Too few dimensions in input data")
    elif len(array.shape) != len(shape_out):
        raise ValueError("Number of dimensions in input and output data should match")

    # Separate the "extra" dimensions that don't correspond to a WCS axis and
    # which we'll be looping over
    extra_dimens_in = array.shape[: -wcs_in.low_level_wcs.pixel_n_dim]
    extra_dimens_out = shape_out[: -wcs_out.low_level_wcs.pixel_n_dim]
    if extra_dimens_in != extra_dimens_out:
        raise ValueError("Dimensions to be looped over must match exactly")

    # TODO: at the moment, we compute the coordinates of all of the corners,
    # but we might want to do it in steps for large images.

    # Start off by finding the world position of all the corners of the input
    # image in world coordinates

    ny_in, nx_in = array.shape[-2:]

    x = np.arange(nx_in + 1.0) - 0.5
    y = np.arange(ny_in + 1.0) - 0.5

    xp_in, yp_in = np.meshgrid(x, y, indexing="xy", sparse=False, copy=False)

    world_in = wcs_in.pixel_to_world(xp_in, yp_in)

    # Now compute the world positions of all the corners in the output header

    ny_out, nx_out = shape_out[-2:]

    x = np.arange(nx_out + 1.0) - 0.5
    y = np.arange(ny_out + 1.0) - 0.5

    xp_out, yp_out = np.meshgrid(x, y, indexing="xy", sparse=False, copy=False)

    world_out = wcs_out.pixel_to_world(xp_out, yp_out)

    # Convert the input world coordinates to the frame of the output world
    # coordinates.

    world_in = world_in.transform_to(world_out.frame)

    # Finally, compute the pixel positions in the *output* image of the pixels
    # from the *input* image.

    xp_inout, yp_inout = wcs_out.world_to_pixel(world_in)

    world_in_unitsph = world_in.represent_as("unitspherical")
    xw_in, yw_in = world_in_unitsph.lon.to_value(u.deg), world_in_unitsph.lat.to_value(u.deg)

    world_out_unitsph = world_out.represent_as("unitspherical")
    xw_out, yw_out = world_out_unitsph.lon.to_value(u.deg), world_out_unitsph.lat.to_value(u.deg)

    # If the input array contains extra dimensions beyond what the input WCS
    # has, the extra leading dimensions are assumed to represent multiple
    # images with the same coordinate information. The transformation is
    # computed once and "broadcast" across those images.
    if len(array.shape) == wcs_in.low_level_wcs.pixel_n_dim:
        # We don't need to broadcast the transformation over any extra
        # axes---add an extra axis of length one just so we have something
        # to loop over in all cases.
        broadcasting = False
        array = array.reshape((1, *array.shape))
    elif len(array.shape) > wcs_in.low_level_wcs.pixel_n_dim:
        # We're broadcasting. Flatten the extra dimensions so there's just one
        # to loop over
        broadcasting = True
        array = array.reshape((-1, *array.shape[-wcs_in.low_level_wcs.pixel_n_dim :]))
        array_out = array_out.reshape((-1, *shape_out[-2:]))
        output_footprint = output_footprint.reshape((-1, *shape_out[-2:]))
    else:
        raise ValueError("Too few dimensions for input array")

    array = np.ascontiguousarray(array)

    for i in range(len(array)):
        array_new, weights = _reproject_slice_cython(
            0,
            nx_in,
            0,
            ny_in,
            nx_out,
            ny_out,
            np.ascontiguousarray(xp_inout),
            np.ascontiguousarray(yp_inout),
            np.ascontiguousarray(xw_in),
            np.ascontiguousarray(yw_in),
            np.ascontiguousarray(xw_out),
            np.ascontiguousarray(yw_out),
            array[i],
            shape_out[-2:],
        )

        with np.errstate(invalid="ignore"):
            array_new /= weights

        if broadcasting:
            array_out[i] = array_new
            if return_footprint:
                output_footprint[i] = weights
        else:
            array_out[:] = array_new
            if return_footprint:
                output_footprint[:] = weights

    if broadcasting:
        array_out = array_out.reshape(shape_out)
        if return_footprint:
            output_footprint = output_footprint.reshape(shape_out)

    if return_footprint:
        return array_out, output_footprint
    else:
        return array_out

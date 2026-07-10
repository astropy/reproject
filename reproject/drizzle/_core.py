# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np


def _reproject_drizzle(
    array,
    wcs_in,
    wcs_out,
    shape_out,
    kernel="square",
    pixfrac=1.0,
    array_out=None,
    output_footprint=None,
    return_footprint=True,
):
    try:
        from drizzle.resample import Drizzle
        from drizzle.utils import calc_pixmap, estimate_pixel_scale_ratio
    except ImportError:
        raise ImportError(
            "The drizzle package is required to use reproject_drizzle and can "
            "be installed with 'pip install drizzle'"
        ) from None

    if array_out is None:
        array_out = np.empty(shape_out)

    if output_footprint is None:
        output_footprint = np.empty(shape_out)

    shape_out = tuple(shape_out)

    if wcs_in.pixel_n_dim != 2:
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

    low_level_wcs_in = getattr(wcs_in, "low_level_wcs", wcs_in)
    low_level_wcs_out = getattr(wcs_out, "low_level_wcs", wcs_out)

    # Map the center of every input pixel to its position in the output image.
    # This mapping is all the drizzle package needs from the WCS, and only
    # uses the APE 14 *_values methods, so any WCS reproject accepts works.
    pixmap = calc_pixmap(low_level_wcs_in, low_level_wcs_out, shape=array.shape[-2:])

    # The weight map that drizzle accumulates is the overlap area in units of
    # *input* pixel areas, whereas the reproject footprint convention is the
    # fraction of each output pixel covered by valid input values. The two
    # differ by the output-to-input pixel area ratio, which we estimate at the
    # image centers (so the footprint normalization is approximate for images
    # with strong distortion or large projection-induced scale variation).
    scale_ratio = estimate_pixel_scale_ratio(low_level_wcs_in, low_level_wcs_out)

    # If the input array contains extra dimensions beyond what the input WCS
    # has, the extra leading dimensions are assumed to represent multiple
    # images with the same coordinate information. The pixel mapping is
    # computed once and "broadcast" across those images.
    if len(array.shape) == wcs_in.low_level_wcs.pixel_n_dim:
        # We don't need to broadcast the transformation over any extra
        # axes---add an extra axis of length one just so we have something
        # to loop over in all cases.
        array = array.reshape((1, *array.shape))
        array_out_loopable = array_out.reshape((1, *shape_out[-2:]))
        footprint_loopable = output_footprint.reshape((1, *shape_out[-2:]))
    elif len(array.shape) > wcs_in.low_level_wcs.pixel_n_dim:
        # We're broadcasting. Flatten the extra dimensions so there's just one
        # to loop over
        array = array.reshape((-1, *array.shape[-2:]))
        array_out_loopable = array_out.reshape((-1, *shape_out[-2:]))
        footprint_loopable = output_footprint.reshape((-1, *shape_out[-2:]))
    else:
        raise ValueError("Too few dimensions for input array")

    for i in range(len(array)):
        driz = Drizzle(kernel=kernel, out_shape=shape_out[-2:], fillval=np.nan, disable_ctx=True)
        driz.add_image(np.asarray(array[i]), exptime=1.0, pixmap=pixmap, pixfrac=pixfrac)
        array_out_loopable[i] = driz.out_img
        footprint_loopable[i] = driz.out_wht / scale_ratio**2

    if return_footprint:
        return array_out, output_footprint
    else:
        return array_out

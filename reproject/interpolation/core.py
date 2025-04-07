# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_pixel

from ..array_utils import map_coordinates
from ..wcs_utils import has_celestial, pixel_to_pixel_with_roundtrip


def _validate_wcs(wcs_in, wcs_out, shape_in, shape_out):
    if wcs_in.low_level_wcs.pixel_n_dim != wcs_out.low_level_wcs.pixel_n_dim:
        raise ValueError("Number of dimensions in input and output WCS should match")
    elif len(shape_out) < wcs_out.low_level_wcs.pixel_n_dim:
        raise ValueError("Too few dimensions in shape_out")
    elif len(shape_in) < wcs_in.low_level_wcs.pixel_n_dim:
        raise ValueError("Too few dimensions in input data")
    elif len(shape_in) != len(shape_out):
        raise ValueError("Number of dimensions in input and output data should match")

    # Separate the "extra" dimensions that don't correspond to a WCS axis and
    # which we'll be looping over
    extra_dimens_in = shape_in[: -wcs_in.low_level_wcs.pixel_n_dim]
    extra_dimens_out = shape_out[: -wcs_out.low_level_wcs.pixel_n_dim]
    if extra_dimens_in != extra_dimens_out:
        raise ValueError("Dimensions to be looped over must match exactly")

    if has_celestial(wcs_in) and not has_celestial(wcs_out):
        raise ValueError("Input WCS has celestial components but output WCS does not")
    elif has_celestial(wcs_out) and not has_celestial(wcs_in):
        raise ValueError("Output WCS has celestial components but input WCS does not")

    if isinstance(wcs_in, WCS) and isinstance(wcs_out, WCS):
        # Check whether a spectral component is present, and if so, check that
        # the CTYPEs match.
        if wcs_in.wcs.spec >= 0 and wcs_out.wcs.spec >= 0:
            if wcs_in.wcs.ctype[wcs_in.wcs.spec] != wcs_out.wcs.ctype[wcs_out.wcs.spec]:
                raise ValueError(
                    f"The input ({wcs_in.wcs.ctype[wcs_in.wcs.spec]}) and output ({wcs_out.wcs.ctype[wcs_out.wcs.spec]}) spectral "
                    "coordinate types are not equivalent."
                )
        elif wcs_in.wcs.spec >= 0:
            raise ValueError("Input WCS has a spectral component but output WCS does not")
        elif wcs_out.wcs.spec >= 0:
            raise ValueError("Output WCS has a spectral component but input WCS does not")


def _reproject_full(
    array,
    wcs_in,
    wcs_out,
    shape_out,
    order=1,
    array_out=None,
    return_footprint=True,
    roundtrip_coords=True,
    output_footprint=None,
):
    """
    Reproject n-dimensional data to a new projection using interpolation.

    The input and output WCS and shape have to satisfy a number of conditions:

    - The number of dimensions in each WCS should match
    - The output shape should match the dimensionality of the WCS
    - The input and output WCS should have matching physical types, although
      the order can be different as long as the physical types are unique.

    If the input array contains extra dimensions beyond what the input WCS has,
    the extra leading dimensions are assumed to represent multiple images with
    the same coordinate information. The transformation is computed once and
    "broadcast" across those images.
    """

    # shape_out must be exactly a tuple type
    shape_out = tuple(shape_out)
    _validate_wcs(wcs_in, wcs_out, array.shape, shape_out)

    if array_out is None:
        array_out = np.empty(shape_out)

    if output_footprint is None:
        output_footprint = np.empty(shape_out)

    array_out_loopable = array_out
    if len(array.shape) == wcs_in.low_level_wcs.pixel_n_dim:
        # We don't need to broadcast the transformation over any extra
        # axes---add an extra axis of length one just so we have something
        # to loop over in all cases.
        array = array.reshape((1, *array.shape))
        array_out_loopable = array_out.reshape((1, *array_out.shape))
    elif len(array.shape) > wcs_in.low_level_wcs.pixel_n_dim:
        # We're broadcasting. Flatten the extra dimensions so there's just one
        # to loop over
        array = array.reshape((-1, *array.shape[-wcs_in.low_level_wcs.pixel_n_dim :]))
        array_out_loopable = array_out.reshape(
            (-1, *array_out.shape[-wcs_out.low_level_wcs.pixel_n_dim :])
        )
    else:
        raise ValueError("Too few dimensions for input array")

    wcs_dims = shape_out[-wcs_in.low_level_wcs.pixel_n_dim :]
    pixel_out = np.meshgrid(
        *[np.arange(size, dtype=float) for size in wcs_dims],
        indexing="ij",
        sparse=False,
        copy=False,
    )
    pixel_out = [p.ravel() for p in pixel_out]
    # For each pixel in the output array, get the pixel value in the input WCS
    if roundtrip_coords:
        pixel_in = pixel_to_pixel_with_roundtrip(wcs_out, wcs_in, *pixel_out[::-1])[::-1]
    else:
        pixel_in = pixel_to_pixel(wcs_out, wcs_in, *pixel_out[::-1])[::-1]
    pixel_in = np.array(pixel_in)

    # Loop over the broadcasted dimensions in our array, reusing the same
    # computed transformation each time
    for i in range(len(array)):
        # Interpolate array on to the pixels coordinates in pixel_in
        map_coordinates(
            array[i],
            pixel_in,
            order=order,
            cval=np.nan,
            mode="constant",
            output=array_out_loopable[i].ravel(),
            max_chunk_size=256 * 1024**2,
        )

    # n.b. We write the reprojected data into array_out_loopable, but array_out
    # also contains this data and has the user's desired output shape.

    if return_footprint:
        output_footprint[:] = (~np.isnan(array_out)).astype(float)
        return array_out, output_footprint
    else:
        return array_out

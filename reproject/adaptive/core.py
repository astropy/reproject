# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from astropy.wcs.utils import pixel_to_pixel

from ..wcs_utils import pixel_to_pixel_with_roundtrip
from .deforest import map_coordinates

__all__ = ["_reproject_adaptive_2d"]


class CoordinateTransformer:
    def __init__(self, wcs_in, wcs_out, roundtrip_coords):
        self.wcs_in = wcs_in
        self.wcs_out = wcs_out
        self.roundtrip_coords = roundtrip_coords

    def __call__(self, pixel_out):
        pixel_out = pixel_out[:, :, 0], pixel_out[:, :, 1]
        if self.roundtrip_coords:
            pixel_in = pixel_to_pixel_with_roundtrip(self.wcs_out, self.wcs_in, *pixel_out)
        else:
            pixel_in = pixel_to_pixel(self.wcs_out, self.wcs_in, *pixel_out)
        pixel_in = np.array(pixel_in).transpose().swapaxes(0, 1)
        return pixel_in


def _reproject_adaptive_2d(
    array,
    wcs_in,
    wcs_out,
    shape_out,
    array_out=None,
    output_footprint=None,
    return_footprint=True,
    center_jacobian=False,
    despike_jacobian=False,
    roundtrip_coords=True,
    conserve_flux=False,
    kernel="Gaussian",
    kernel_width=1.3,
    sample_region_width=4,
    boundary_mode="strict",
    boundary_fill_value=0,
    boundary_ignore_threshold=0.5,
    x_cyclic=False,
    y_cyclic=False,
    bad_value_mode="strict",
    bad_fill_value=0,
):
    """
    Reproject celestial slices from an n-d array from one WCS to another
    using the DeForest (2004) algorithm [1]_, and assuming all other dimensions
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
    return_footprint : bool
        Whether to return the footprint in addition to the output array.
    center_jacobian : bool
        Whether to compute centered Jacobians
    despike_jacobian : bool
        Whether to despike the Jacobians
    roundtrip_coords : bool
        Whether to veryfiy that coordinate transformations are defined in both
        directions.
    conserve_flux : bool
        Whether to rescale output pixel values so flux is conserved
    kernel : str
        The averaging kernel to use.
    kernel_width : double
        The width of the kernel in pixels. Applies only to the Gaussian kernel.
    sample_region_width : double
        The width in pixels of the sample region, used only for the Gaussian
        kernel which otherwise has infinite extent.
    boundary_mode : str
        Boundary handling mode
    boundary_fill_value : double
        Fill value for 'constant' boundary mode
    boundary_ignore_threshold : double
        Threshold for 'ignore_threshold' boundary mode, ranging from 0 to 1.
    x_cyclic, y_cyclic : bool
        Marks in input-image axis as cyclic.
    bad_value_mode : str
        NaN and inf handling mode
    bad_fill_value : float
        Fill value for 'constant' bad value mode

    Returns
    -------
    array_new : `~numpy.ndarray`
        The reprojected array
    footprint : `~numpy.ndarray`
        Footprint of the input array in the output array. Values of 0 indicate
        no coverage or valid values in the input image, while values of 1
        indicate valid values.

    References
    ----------
    .. [1] C. E. DeForest, "On Re-sampling of Solar Images"
       Solar Physics volume 219, pages 3–23 (2004),
       https://doi.org/10.1023/B:SOLA.0000021743.24248.b0

    Note
    ----
    If the input array contains extra dimensions beyond what the input WCS has,
    the extra leading dimensions are assumed to represent multiple images with
    the same coordinate information. The transformation is computed once and
    "broadcast" across those images.
    """

    # Make sure image is floating point
    array_in = np.asarray(array, dtype=float)
    shape_out = tuple(shape_out)

    # Check dimensionality of WCS and shape_out
    if wcs_in.low_level_wcs.pixel_n_dim != wcs_out.low_level_wcs.pixel_n_dim:
        raise ValueError("Number of dimensions between input and output WCS should match")
    elif len(shape_out) < wcs_out.low_level_wcs.pixel_n_dim:
        raise ValueError("Too few dimensions in shape_out")
    elif len(array_in.shape) < wcs_in.low_level_wcs.pixel_n_dim:
        raise ValueError("Too few dimensions in input data")
    elif len(array_in.shape) != len(shape_out):
        raise ValueError("Number of dimensions in input and output data should match")

    # Separate the "extra" dimensions that don't correspond to a WCS axis and
    # which we'll be looping over
    extra_dimens_in = array_in.shape[: -wcs_in.low_level_wcs.pixel_n_dim]
    extra_dimens_out = shape_out[: -wcs_out.low_level_wcs.pixel_n_dim]
    if extra_dimens_in != extra_dimens_out:
        raise ValueError("Dimensions to be looped over must match exactly")

    if array_out is None:
        array_out = np.empty(shape_out)

    if output_footprint is None:
        output_footprint = np.empty(shape_out)

    if len(array_in.shape) == wcs_in.low_level_wcs.pixel_n_dim:
        # We don't need to broadcast the transformation over any extra
        # axes---add an extra axis of length one just so we have something
        # to loop over in all cases.
        array_in = array_in.reshape((1, *array_in.shape))
        array_out = array_out.reshape((1, *array_out.shape))
    elif len(array_in.shape) > wcs_in.low_level_wcs.pixel_n_dim:
        # We're broadcasting. Flatten the extra dimensions so there's just one
        # to loop over
        array_in = array_in.reshape((-1, *array_in.shape[-wcs_in.low_level_wcs.pixel_n_dim :]))
        array_out = array_out.reshape((-1, *array_out.shape[-wcs_out.low_level_wcs.pixel_n_dim :]))

    transformer = CoordinateTransformer(wcs_in, wcs_out, roundtrip_coords)
    map_coordinates(
        array_in,
        array_out,
        transformer,
        out_of_range_nan=True,
        center_jacobian=center_jacobian,
        despiked_jacobian=despike_jacobian,
        conserve_flux=conserve_flux,
        kernel=kernel,
        kernel_width=kernel_width,
        sample_region_width=sample_region_width,
        boundary_mode=boundary_mode,
        boundary_fill_value=boundary_fill_value,
        boundary_ignore_threshold=boundary_ignore_threshold,
        x_cyclic=x_cyclic,
        y_cyclic=y_cyclic,
        bad_value_mode=bad_value_mode,
        bad_fill_value=bad_fill_value,
    )

    array_out.shape = shape_out

    if return_footprint:
        output_footprint[:] = (~np.isnan(array_out)).astype(float)
        return array_out, output_footprint
    else:
        return array_out

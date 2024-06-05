# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from astropy.wcs import WCS
from astropy.wcs.wcsapi import SlicedLowLevelWCS

from ..array_utils import sample_array_edges
from ..utils import parse_input_data, parse_input_weights, parse_output_projection
from .background import determine_offset_matrix, solve_corrections_sgd
from .subset_array import ReprojectedArraySubset

__all__ = ["reproject_and_coadd"]


def _noop(iterable):
    return iterable


def reproject_and_coadd(
    input_data,
    output_projection,
    shape_out=None,
    input_weights=None,
    hdu_in=None,
    hdu_weights=None,
    reproject_function=None,
    combine_function="mean",
    match_background=False,
    background_reference=None,
    output_array=None,
    output_footprint=None,
    block_sizes=None,
    progress_bar=None,
    blank_pixel_value=0,
    **kwargs,
):
    """
    Given a set of input data, reproject and co-add these to a single
    final image.

    Parameters
    ----------
    input_data : iterable
        One or more input datasets to reproject and co-add. This should be an
        iterable containing one entry for each dataset, where a single dataset
        is one of:

            * The name of a FITS file
            * An `~astropy.io.fits.HDUList` object
            * An image HDU object such as a `~astropy.io.fits.PrimaryHDU`,
              `~astropy.io.fits.ImageHDU`, or `~astropy.io.fits.CompImageHDU`
              instance
            * A tuple where the first element is an `~numpy.ndarray` and the
              second element is either a `~astropy.wcs.WCS` or a
              `~astropy.io.fits.Header` object
            * An `~astropy.nddata.NDData` object from which the ``.data`` and
              ``.wcs`` attributes will be used as the input data.

    output_projection : `~astropy.wcs.wcsapi.BaseLowLevelWCS` or `~astropy.wcs.wcsapi.BaseHighLevelWCS` or `~astropy.io.fits.Header`
        The output projection, which can be either a
        `~astropy.wcs.wcsapi.BaseLowLevelWCS`,
        `~astropy.wcs.wcsapi.BaseHighLevelWCS`, or a `~astropy.io.fits.Header`
        instance.
    shape_out : tuple, optional
        If ``output_projection`` is a WCS instance, the shape of the output
        data should be specified separately.
    input_weights : iterable
        If specified, this should be an iterable with the same length as
        ``input_data``, where each item is one of:

            * The name of a FITS file
            * An `~astropy.io.fits.HDUList` object
            * An image HDU object such as a `~astropy.io.fits.PrimaryHDU`,
              `~astropy.io.fits.ImageHDU`, or `~astropy.io.fits.CompImageHDU`
              instance
            * An `~numpy.ndarray` array

    hdu_in : int or str, optional
        If one or more items in ``input_data`` is a FITS file or an
        `~astropy.io.fits.HDUList` instance, specifies the HDU to use.
    hdu_weights : int or str, optional
        If one or more items in ``input_weights`` is a FITS file or an
        `~astropy.io.fits.HDUList` instance, specifies the HDU to use.
    reproject_function : callable
        The function to use for the reprojection.
    combine_function : { 'mean', 'sum', 'first', 'last', 'min', 'max' }
        The type of function to use for combining the values into the final
        image. For 'first' and 'last', respectively, the reprojected images are
        simply overlaid on top of each other. With respect to the order of the
        input images in ``input_data``, either the first or the last image to
        cover a region of overlap determines the output data for that region.
    match_background : bool
        Whether to match the backgrounds of the images.
    background_reference : `None` or `int`
        If `None`, the background matching will make it so that the average of
        the corrections for all images is zero. If an integer, this specifies
        the index of the image to use as a reference.
    output_array : array or None
        The final output array.  Specify this if you already have an
        appropriately-shaped array to store the data in.  Must match shape
        specified with `shape_out` or derived from the output
        projection.
    output_footprint : array or None
        The final output footprint array.  Specify this if you already have an
        appropriately-shaped array to store the data in.  Must match shape
        specified with `shape_out` or derived from the output projection.
    block_sizes : list of tuples or None
        The block size to use for each dataset.  Could also be a single tuple
        if you want the sample block size for all data sets.
    progress_bar : callable, optional
        If specified, use this as a progress_bar to track loop iterations over
        data sets.
    blank_pixel_value : float, optional
        Value to use for areas of the resulting mosaic that do not have input
        data.

    **kwargs
        Keyword arguments to be passed to the reprojection function.

    Returns
    -------
    array : `~numpy.ndarray`
        The co-added array.
    footprint : `~numpy.ndarray`
        Footprint of the co-added array. Values of 0 indicate no coverage or
        valid values in the input image, while values of 1 indicate valid
        values.
    """

    # TODO: add support for saving intermediate files to disk to avoid blowing
    # up memory usage. We could probably still have references to array
    # objects, but we'd just make sure these were memory mapped

    # Validate inputs

    if combine_function not in ("mean", "sum", "first", "last", "min", "max"):
        raise ValueError("combine_function should be one of mean/sum/first/last/min/max")

    if reproject_function is None:
        raise ValueError(
            "reprojection function should be specified with the reproject_function argument"
        )

    if progress_bar is None:
        progress_bar = _noop

    # Parse the output projection to avoid having to do it for each

    wcs_out, shape_out = parse_output_projection(output_projection, shape_out=shape_out)

    if output_array is None:
        output_array = np.zeros(shape_out)
    elif output_array.shape != shape_out:
        raise ValueError(
            "If you specify an output array, it must have a shape matching "
            f"the output shape {shape_out}"
        )

    if output_footprint is None:
        output_footprint = np.zeros(shape_out)
    elif output_footprint.shape != shape_out:
        raise ValueError(
            "If you specify an output footprint array, it must have a shape matching "
            f"the output shape {shape_out}"
        )

    # Define 'on-the-fly' mode: in the case where we don't need to match
    # the backgrounds and we are combining with 'mean' or 'sum', we don't
    # have to keep track of the intermediate arrays and can just modify
    # the output array on-the-fly
    on_the_fly = not match_background and combine_function in ("mean", "sum")

    # Start off by reprojecting individual images to the final projection

    if not on_the_fly:
        arrays = []

    for idata in progress_bar(range(len(input_data))):
        # We need to pre-parse the data here since we need to figure out how to
        # optimize/minimize the size of each output tile (see below).
        array_in, wcs_in = parse_input_data(input_data[idata], hdu_in=hdu_in)

        # We also get the weights map, if specified
        if input_weights is None:
            weights_in = None
        else:
            weights_in = parse_input_weights(input_weights[idata], hdu_weights=hdu_weights)
            if np.any(np.isnan(weights_in)):
                weights_in = np.nan_to_num(weights_in)

        # Since we might be reprojecting small images into a large mosaic we
        # want to make sure that for each image we reproject to an array with
        # minimal footprint. We therefore find the pixel coordinates of the
        # edges of the initial image and transform this to pixel coordinates in
        # the final image to figure out the final WCS and shape to reproject to
        # for each tile. We strike a balance between transforming only the
        # input-image corners, which is fast but can cause clipping in cases of
        # significant distortion (when the edges of the input image become
        # convex in the output projection), and transforming every edge pixel,
        # which provides a lot of redundant information.

        edges = sample_array_edges(array_in.shape, n_samples=11)[::-1]
        edges_out = wcs_out.world_to_pixel(wcs_in.pixel_to_world(*edges))[::-1]

        # Determine the cutout parameters

        # In some cases, images might not have valid coordinates in the corners,
        # such as all-sky images or full solar disk views. In this case we skip
        # this step and just use the full output WCS for reprojection.

        ndim_out = len(shape_out)

        skip_data = False
        if np.any(np.isnan(edges_out)):
            bounds = list(zip([0] * ndim_out, shape_out))
        else:
            bounds = []
            for idim in range(ndim_out):
                imin = max(0, int(np.floor(edges_out[idim].min() + 0.5)))
                imax = min(shape_out[idim], int(np.ceil(edges_out[idim].max() + 0.5)))
                bounds.append((imin, imax))
                if imax < imin:
                    skip_data = True
                    break

        if skip_data:
            continue

        slice_out = tuple([slice(imin, imax) for (imin, imax) in bounds])

        if isinstance(wcs_out, WCS):
            wcs_out_indiv = wcs_out[slice_out]
        else:
            wcs_out_indiv = SlicedLowLevelWCS(wcs_out.low_level_wcs, slice_out)

        shape_out_indiv = [imax - imin for (imin, imax) in bounds]

        if block_sizes is not None:
            if len(block_sizes) == len(input_data) and len(block_sizes[idata]) == len(shape_out):
                kwargs["block_size"] = block_sizes[idata]
            else:
                kwargs["block_size"] = block_sizes

        # TODO: optimize handling of weights by making reprojection functions
        # able to handle weights, and make the footprint become the combined
        # footprint + weight map

        array, footprint = reproject_function(
            (array_in, wcs_in),
            output_projection=wcs_out_indiv,
            shape_out=shape_out_indiv,
            hdu_in=hdu_in,
            **kwargs,
        )

        if weights_in is not None:
            weights, _ = reproject_function(
                (weights_in, wcs_in),
                output_projection=wcs_out_indiv,
                shape_out=shape_out_indiv,
                hdu_in=hdu_in,
                **kwargs,
            )

        # For the purposes of mosaicking, we mask out NaN values from the array
        # and set the footprint to 0 at these locations.
        reset = np.isnan(array)
        array[reset] = 0.0
        footprint[reset] = 0.0

        # Combine weights and footprint
        if weights_in is not None:
            weights[reset] = 0.0
            footprint *= weights

        array = ReprojectedArraySubset(array, footprint, bounds)

        # TODO: make sure we gracefully handle the case where the
        # output image is empty (due e.g. to no overlap).

        if on_the_fly:
            # By default, values outside of the footprint are set to NaN
            # but we set these to 0 here to avoid getting NaNs in the
            # means/sums.
            array.array[array.footprint == 0] = 0
            output_array[array.view_in_original_array] += array.array * array.footprint
            output_footprint[array.view_in_original_array] += array.footprint
        else:
            arrays.append(array)

    # If requested, try and match the backgrounds.
    if match_background and len(arrays) > 1:
        offset_matrix = determine_offset_matrix(arrays)
        corrections = solve_corrections_sgd(offset_matrix)
        if background_reference:
            corrections -= corrections[background_reference]
        for array, correction in zip(arrays, corrections, strict=True):
            array.array -= correction

    if combine_function in ("mean", "sum"):
        if match_background:
            # if we're not matching the background, this part has already been done
            for array in arrays:
                # By default, values outside of the footprint are set to NaN
                # but we set these to 0 here to avoid getting NaNs in the
                # means/sums.
                array.array[array.footprint == 0] = 0

                output_array[array.view_in_original_array] += array.array * array.footprint
                output_footprint[array.view_in_original_array] += array.footprint

        if combine_function == "mean":
            with np.errstate(invalid="ignore"):
                output_array /= output_footprint
                output_array[output_footprint == 0] = blank_pixel_value

    elif combine_function in ("first", "last", "min", "max"):
        if combine_function == "min":
            output_array[...] = np.inf
        elif combine_function == "max":
            output_array[...] = -np.inf

        for array in arrays:
            if combine_function == "first":
                mask = output_footprint[array.view_in_original_array] == 0
            elif combine_function == "last":
                mask = array.footprint > 0
            elif combine_function == "min":
                mask = (array.footprint > 0) & (
                    array.array < output_array[array.view_in_original_array]
                )
            elif combine_function == "max":
                mask = (array.footprint > 0) & (
                    array.array > output_array[array.view_in_original_array]
                )

            output_footprint[array.view_in_original_array] = np.where(
                mask, array.footprint, output_footprint[array.view_in_original_array]
            )
            output_array[array.view_in_original_array] = np.where(
                mask, array.array, output_array[array.view_in_original_array]
            )

    output_array[output_footprint == 0] = blank_pixel_value

    return output_array, output_footprint

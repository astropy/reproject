# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os
import sys
import tempfile
import uuid
from logging import getLogger

import dask.array as da
import numpy as np
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_pixel
from astropy.wcs.wcsapi import SlicedLowLevelWCS

from .._array_utils import iterate_chunks, sample_array_edges
from ..interpolation._core import _validate_wcs
from ..utils import parse_input_data, parse_input_weights, parse_output_projection
from ._background import determine_offset_matrix, solve_corrections_sgd
from ._subset_array import DEFAULT_MAX_CHUNK_SIZE, ReprojectedArraySubset

__all__ = ["reproject_and_coadd"]


IS_WIN = sys.platform == "win32"


def _noop(iterable):
    return iterable


def _safe_remove(path):
    try:
        os.remove(path)
    except PermissionError:
        pass


def _combine_array_into_output(combine_function, array, output_array, output_footprint):
    for chunk in array.as_chunks():
        # Values outside of the footprint are set to NaN by default
        # but we set these to 0 here to avoid NaNs in the means/sums.
        if combine_function in ("mean", "sum"):
            chunk.array[chunk.footprint == 0] = 0.0
            output_footprint[chunk.view_in_original_array] += chunk.footprint
            output_array[chunk.view_in_original_array] += chunk.array * chunk.footprint
        elif combine_function in ("first", "last", "min", "max"):
            if combine_function == "first":
                mask = (chunk.footprint > 0) & (output_footprint[chunk.view_in_original_array] == 0)
            elif combine_function == "last":
                mask = chunk.footprint > 0
            elif combine_function == "min":
                mask = (chunk.footprint > 0) & (
                    chunk.array < output_array[chunk.view_in_original_array]
                )
            elif combine_function == "max":
                mask = (chunk.footprint > 0) & (
                    chunk.array > output_array[chunk.view_in_original_array]
                )

            # Update only the selected pixels in place, which avoids allocating
            # and rewriting the whole chunk as np.where would.
            np.copyto(output_footprint[chunk.view_in_original_array], chunk.footprint, where=mask)
            np.copyto(output_array[chunk.view_in_original_array], chunk.array, where=mask)
        else:
            raise ValueError(f"Unexpected combine_function: {combine_function}")


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
    non_reprojected_dims=None,
    progress_bar=None,
    blank_pixel_value=0,
    intermediate_memmap=False,
    return_type=None,
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
        specified with ``shape_out`` or derived from the output
        projection.
    output_footprint : array or None
        The final output footprint array.  Specify this if you already have an
        appropriately-shaped array to store the data in.  Must match shape
        specified with ``shape_out`` or derived from the output projection.
    block_sizes : list of tuples or None
        The block size to use for each dataset.  Could also be a single tuple
        if you want the same block size for all data sets.
    non_reprojected_dims : tuple, optional
        Leading dimensions of the data that should not be reprojected but for
        which a one-to-one mapping between input and output pixels is assumed
        (see the ``reproject_interp`` documentation for details). This makes it
        possible to co-add cubes where the input and output WCS have the same
        number of dimensions as the data, broadcasting the co-addition over
        these dimensions. It is passed through to ``reproject_function``.
    progress_bar : callable, optional
        If specified, use this as a progress_bar to track loop iterations over
        data sets.
    blank_pixel_value : float, optional
        Value to use for areas of the resulting mosaic that do not have input
        data.
    intermediate_memmap : bool, optional
        If `True`, use `numpy.memmap` to store intermediate output arrays for
        reprojected data.
    return_type : {None, 'dask'}, optional
        If ``'dask'``, reproject each image lazily (using ``return_type='dask'``
        on the reprojection function), pad each onto the output grid, and combine
        them with a single nan-aware reduction, returning the resulting
        **uncomputed** dask arrays so the whole co-addition is deferred into one
        computation. This currently supports ``combine_function`` of 'mean',
        'sum', 'min' or 'max' (each weighting the images equally rather than by
        footprint), and does not support ``match_background`` or
        ``input_weights``. The default (`None`) uses the standard eager
        co-addition.

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

    logger = getLogger(__name__)

    # Validate inputs

    if combine_function not in ("mean", "sum", "first", "last", "min", "max"):
        raise ValueError("combine_function should be one of mean/sum/first/last/min/max")

    if reproject_function is None:
        raise ValueError(
            "reprojection function should be specified with the reproject_function argument"
        )

    if "block_size" in kwargs and kwargs["block_size"] is not None:
        if block_sizes is not None:
            raise ValueError("Cannot specify block_sizes= and block_size= at the same time")
        block_sizes = kwargs.pop("block_size")

    # non_reprojected_dims is used below to size the cutouts correctly and is
    # also forwarded to reproject_function for each individual reprojection.
    if non_reprojected_dims is not None:
        kwargs["non_reprojected_dims"] = non_reprojected_dims

    if progress_bar is None:
        progress_bar = _noop

    # Parse the output projection to avoid having to do it for each

    wcs_out, shape_out = parse_output_projection(output_projection, shape_out=shape_out)

    # Extracting per-image cutouts below slices the output WCS, which requires it
    # to know its array shape. Older astropy versions do not infer this for WCS
    # objects created without one (so slicing an N-d WCS raises an IndexError),
    # so set it explicitly here on a copy to avoid mutating the user's WCS.
    if isinstance(wcs_out, WCS) and wcs_out.array_shape is None:
        wcs_out = wcs_out.deepcopy()
        wcs_out.array_shape = shape_out[-wcs_out.low_level_wcs.pixel_n_dim :]

    # When return_type='dask', we build the entire co-addition as a single deferred
    # dask graph: each image is reprojected lazily, padded onto the output grid, and
    # the images are combined with a nan-aware reduction. The uncomputed dask arrays
    # are returned so the whole thing is computed once at the end, so we must not
    # allocate the (potentially huge) output arrays here.
    coadd_with_dask = return_type == "dask"

    if coadd_with_dask:
        if match_background:
            raise ValueError("Cannot use return_type='dask' together with match_background")
        if input_weights is not None:
            raise NotImplementedError("return_type='dask' does not yet support input_weights")
        if combine_function not in ("mean", "sum", "min", "max"):
            raise NotImplementedError(
                "return_type='dask' currently only supports combine_function 'mean', "
                "'sum', 'min' or 'max'"
            )
        dask_arrays = []
    else:
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

    logger.info(f"Output mosaic will have shape {shape_out}")

    # Define 'on-the-fly' mode: in the case where we don't need to match the
    # backgrounds, we don't have to keep track of the intermediate arrays and
    # can just modify the output array on-the-fly
    on_the_fly = not match_background

    on_the_fly_prefix = "Using" if on_the_fly else "Not using"
    logger.info(
        f"{on_the_fly_prefix} on-the-fly mode for adding individual reprojected images to output array"
    )

    # Start off by reprojecting individual images to the final projection

    if not on_the_fly:
        arrays = []

    if not coadd_with_dask:
        if combine_function == "min":
            output_array[...] = np.inf
        elif combine_function == "max":
            output_array[...] = -np.inf

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=IS_WIN) as local_tmp_dir:

        for idata in progress_bar(range(len(input_data))):

            logger.info(f"Processing input data {idata + 1} of {len(input_data)}")

            # We need to pre-parse the data here since we need to figure out how to
            # optimize/minimize the size of each output tile (see below).
            array_in, wcs_in = parse_input_data(input_data[idata], hdu_in=hdu_in)

            # We also get the weights map, if specified
            if input_weights is None:
                weights_in = None
            else:
                weights_in, weights_wcs = parse_input_weights(
                    input_weights[idata], hdu_weights=hdu_weights
                )
                if weights_wcs is None:
                    # if weights are passed as an array
                    weights_wcs = wcs_in
                else:
                    try:
                        _validate_wcs(weights_wcs, wcs_in, weights_in.shape, shape_out)
                    except ValueError:
                        # WCS is not valid (most likely, it is blank?)
                        weights_wcs = wcs_in
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

            try:
                edges = sample_array_edges(
                    array_in.shape[-wcs_in.low_level_wcs.pixel_n_dim :], n_samples=11
                )[::-1]
                edges_out = pixel_to_pixel(wcs_in, wcs_out, *edges)[::-1]
            except Exception:
                # If the edge coordinates cannot be transformed (for example if
                # they fall outside the validity region of the WCS), fall back to
                # assuming no predicted overlap so the full output is considered.
                edges_out = np.array([np.nan])

            # Determine the cutout parameters

            # In some cases, images might not have valid coordinates in the corners,
            # such as all-sky images or full solar disk views. In this case we skip
            # this step and just use the full output WCS for reprojection.

            ndim_out = len(shape_out)

            # Determine how many leading dimensions are broadcasted (i.e. not
            # reprojected). If non_reprojected_dims is set these are given
            # explicitly; otherwise they are the dimensions for which the output
            # WCS has fewer pixel dimensions than the data.
            if non_reprojected_dims is not None:
                n_broadcasted = len(non_reprojected_dims)
            else:
                n_broadcasted = len(shape_out) - wcs_out.low_level_wcs.pixel_n_dim
            n_dim_reproject = len(shape_out) - n_broadcasted
            # Number of leading dimensions that the output WCS does not itself
            # describe and which therefore have to be skipped when slicing it.
            n_wcs_out_extra = len(shape_out) - wcs_out.low_level_wcs.pixel_n_dim

            skip_data = False
            if np.any(np.isnan(edges_out)):
                bounds = list(zip([0] * ndim_out, shape_out, strict=False))
            else:
                bounds = []
                if n_broadcasted > 0:
                    for idim in range(n_broadcasted):
                        bounds.append((0, shape_out[idim]))
                # Only the reprojected (trailing) dimensions get a cutout; the
                # corresponding edge coordinates are the trailing components of
                # edges_out (which has one entry per input WCS pixel dimension).
                edges_out_reproject = edges_out[-n_dim_reproject:]
                for idim in range(len(edges_out_reproject)):
                    imin = max(0, int(np.floor(edges_out_reproject[idim].min() + 0.5)))
                    imax = min(
                        shape_out[n_broadcasted + idim],
                        int(np.ceil(edges_out_reproject[idim].max() + 0.5)),
                    )
                    bounds.append((imin, imax))
                    if imax <= imin:
                        skip_data = True
                        break

            if skip_data:
                logger.info(
                    "Skipping reprojection as no predicted overlap with final mosaic header"
                )
                continue

            slice_out = tuple([slice(imin, imax) for (imin, imax) in bounds[n_wcs_out_extra:]])

            if isinstance(wcs_out, WCS):
                wcs_out_indiv = wcs_out[slice_out]
            else:
                wcs_out_indiv = SlicedLowLevelWCS(wcs_out.low_level_wcs, slice_out)

            shape_out_indiv = tuple([imax - imin for (imin, imax) in bounds])

            if block_sizes is not None:
                if len(block_sizes) == len(input_data) and len(block_sizes[idata]) == len(
                    shape_out
                ):
                    global_block_size = block_sizes[idata]
                else:
                    global_block_size = block_sizes
            else:
                global_block_size = None

            # If the block size matches the output shape along the reprojected
            # dimensions, we need to shrink those entries to match this cutout's
            # size (keeping the broadcasted entries) so that the per-image
            # reprojection parallelizes over the broadcasted dimensions.
            if global_block_size and tuple(global_block_size[-n_dim_reproject:]) == tuple(
                shape_out[-n_dim_reproject:]
            ):
                block_size = tuple(global_block_size[:n_broadcasted]) + tuple(
                    shape_out_indiv[n_broadcasted:]
                )
            else:
                block_size = global_block_size

            if coadd_with_dask:
                # Reproject this image lazily and pad the cutout back onto the full
                # output grid (filling the uncovered region with NaN so the nan-aware
                # reduction later ignores it). Nothing is computed here.
                logger.info(
                    f"Calling {reproject_function.__name__} lazily with "
                    f"shape_out={shape_out_indiv} (return_type='dask')"
                )
                array = reproject_function(
                    (array_in, wcs_in),
                    output_projection=wcs_out_indiv,
                    shape_out=shape_out_indiv,
                    hdu_in=hdu_in,
                    return_footprint=False,
                    return_type="dask",
                    block_size=block_size,
                    **kwargs,
                )
                pad = [(imin, shape_out[idim] - imax) for idim, (imin, imax) in enumerate(bounds)]
                dask_arrays.append(da.pad(array, pad, constant_values=np.nan))
                continue

            # TODO: optimize handling of weights by making reprojection functions
            # able to handle weights, and make the footprint become the combined
            # footprint + weight map

            if intermediate_memmap:

                array_path = os.path.join(local_tmp_dir, f"array_{uuid.uuid4()}.np")

                logger.info(
                    f"Creating memory-mapped array with shape {shape_out_indiv} at {array_path}"
                )

                array = np.memmap(
                    array_path,
                    shape=shape_out_indiv,
                    mode="w+",
                    dtype=array_in.dtype,
                )

                footprint_path = os.path.join(local_tmp_dir, f"footprint_{uuid.uuid4()}.np")

                logger.info(
                    f"Creating memory-mapped footprint with shape {shape_out_indiv} at {footprint_path}"
                )

                footprint = np.memmap(
                    footprint_path,
                    shape=shape_out_indiv,
                    mode="w+",
                    dtype=float,
                )

            else:

                array = footprint = None

            logger.info(f"Calling {reproject_function.__name__} with shape_out={shape_out_indiv}")

            array, footprint = reproject_function(
                (array_in, wcs_in),
                output_projection=wcs_out_indiv,
                shape_out=shape_out_indiv,
                hdu_in=hdu_in,
                output_array=array,
                output_footprint=footprint,
                block_size=block_size,
                **kwargs,
            )

            if weights_in is not None:

                if intermediate_memmap:

                    weights_path = os.path.join(local_tmp_dir, f"weights_{uuid.uuid4()}.np")

                    logger.info(
                        f"Creating memory-mapped weights with shape {shape_out_indiv} at {weights_path}"
                    )

                    weights = np.memmap(
                        weights_path,
                        shape=shape_out_indiv,
                        mode="w+",
                        dtype=float,
                    )

                else:

                    weights = None

                logger.info(
                    f"Calling {reproject_function.__name__} with shape_out={shape_out_indiv} for weights"
                )

                weights = reproject_function(
                    (weights_in, weights_wcs),
                    output_projection=wcs_out_indiv,
                    shape_out=shape_out_indiv,
                    hdu_in=hdu_in,
                    output_array=weights,
                    return_footprint=False,
                    **kwargs,
                )

            # For the purposes of mosaicking, we mask out NaN values from the array
            # and set the footprint to 0 at these locations. We do this in chunks
            # to avoid excessive memory usage.
            for chunk in iterate_chunks(array.shape, max_chunk_size=DEFAULT_MAX_CHUNK_SIZE):

                # Determine location of NaNs
                reset = np.isnan(array[chunk])
                if weights_in is not None:
                    reset |= np.isnan(weights[chunk])

                # Mask them in-place in the arrays
                array[chunk][reset] = 0.0
                footprint[chunk][reset] = 0.0

                # Combine weights and footprint
                if weights_in is not None:
                    weights[chunk][reset] = 0.0
                    footprint[chunk] *= weights[chunk]

            if weights_in is not None and intermediate_memmap:
                # Remove the reference to the memmap before trying to remove the file itself
                logger.info("Removing memory-mapped weight array")
                weights = None
                _safe_remove(weights_path)

            array = ReprojectedArraySubset(array, footprint, bounds)

            if on_the_fly:
                # Add this reprojected image to the output arrays one chunk at a
                # time. This keeps peak memory usage set by the chunk size rather
                # than by the (potentially large) size of each reprojected image,
                # which matters in particular when there are many non-reprojected
                # dimensions (e.g. spectral channels). Note that these are just
                # chunks over Numpy arrays, not e.g. dask chunks.
                logger.info("Adding reprojected array to final array in chunks")
                _combine_array_into_output(combine_function, array, output_array, output_footprint)

                if intermediate_memmap:
                    logger.info("Removing memory-mapped array and footprint arrays")
                    array = None
                    footprint = None
                    for path in (array_path, footprint_path):
                        _safe_remove(path)

            else:

                logger.info("Adding reprojected array to list to combine later")
                arrays.append(array)

        # If requested, try and match the backgrounds.
        if match_background and len(arrays) > 1:
            logger.info("Match backgrounds")
            offset_matrix = determine_offset_matrix(arrays)
            corrections = solve_corrections_sgd(offset_matrix)
            if background_reference:
                corrections -= corrections[background_reference]
            for array, correction in zip(arrays, corrections, strict=True):
                array.array -= correction

        if match_background:
            logger.info(f"Combining reprojected arrays with function {combine_function}")
            # if we're not matching the background, this part has already been done
            for array in arrays:
                _combine_array_into_output(combine_function, array, output_array, output_footprint)

        if combine_function == "mean" and not coadd_with_dask:
            logger.info("Handle normalization of output array")
            with np.errstate(invalid="ignore"):
                output_array /= output_footprint

        # Avoid keeping any references to the memory-mapped arrays so that the
        # files get cleaned up once we exit the context manager.
        if intermediate_memmap:
            array = None
            footprint = None
            arrays = []

    if coadd_with_dask:
        # Combine all the lazily-reprojected, NaN-padded images with a single
        # nan-aware reduction along the stacking axis. The result is returned
        # uncomputed so the entire graph (reprojections and co-addition) is
        # evaluated in one deferred computation by the caller.
        stacked = da.stack(dask_arrays)
        if combine_function == "mean":
            output_array = da.nanmean(stacked, axis=0)
        elif combine_function == "sum":
            output_array = da.nansum(stacked, axis=0)
        elif combine_function == "min":
            output_array = da.nanmin(stacked, axis=0)
        elif combine_function == "max":
            output_array = da.nanmax(stacked, axis=0)
        output_footprint = da.isfinite(stacked).sum(axis=0)
        return output_array, output_footprint

    # We need to avoid potentially large memory allocation from output == 0 so
    # we operate in chunks.
    logger.info(f"Resetting invalid pixels to {blank_pixel_value}")
    for chunk in iterate_chunks(output_array.shape, max_chunk_size=DEFAULT_MAX_CHUNK_SIZE):
        output_array[chunk][output_footprint[chunk] == 0] = blank_pixel_value

    return output_array, output_footprint

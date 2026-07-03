# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os
import sys
import tempfile
import uuid
from collections import namedtuple
from logging import getLogger

import dask.array as da
import numpy as np
from astropy.wcs import WCS
from astropy.wcs.wcsapi import SlicedLowLevelWCS

from .._array_utils import iterate_chunks, pad_dask_array_to_grid
from ..interpolation._core import _validate_wcs
from ..utils import parse_input_data, parse_input_weights, parse_output_projection
from ._background import determine_offset_matrix, solve_corrections_sgd
from ._subset_array import DEFAULT_MAX_CHUNK_SIZE, ReprojectedArraySubset
from ._wcs_helpers import sample_input_edges_in_output

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


# Everything the per-image reprojection needs to know about one input dataset
# and the cutout of the output grid that it covers.
_InputCutout = namedtuple(
    "_InputCutout",
    [
        "array_in",
        "wcs_in",
        "weights_in",
        "weights_wcs",
        "bounds",
        "wcs_out_indiv",
        "shape_out_indiv",
        "block_size",
    ],
)


def _input_cutout_iterator(
    input_data,
    input_weights,
    hdu_in,
    hdu_weights,
    wcs_out,
    shape_out,
    block_sizes,
    n_broadcasted,
    n_dim_reproject,
    n_wcs_out_extra,
    progress_bar,
):
    # Parse each input dataset (and its weights map) and compute the minimal
    # cutout of the output grid that it covers, yielding everything that the
    # per-image reprojection needs. Datasets with no predicted overlap with the
    # output are skipped. This is shared between the return_type='numpy' and
    # return_type='dask' co-addition drivers so that the cutout logic exists
    # only once.

    logger = getLogger(__name__)

    ndim_out = len(shape_out)

    for idata in progress_bar(range(len(input_data))):

        logger.info(f"Processing input data {idata + 1} of {len(input_data)}")

        # We need to pre-parse the data here since we need to figure out how to
        # optimize/minimize the size of each output tile (see below).
        array_in, wcs_in = parse_input_data(input_data[idata], hdu_in=hdu_in)

        # We also get the weights map, if specified
        if input_weights is None:
            weights_in = None
            weights_wcs = None
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
            edges_out = sample_input_edges_in_output(array_in.shape, wcs_in, wcs_out)
        except Exception as exc:
            # If the edge coordinates cannot be transformed (for example if
            # they fall outside the validity region of the WCS), fall back to
            # assuming no predicted overlap so the full output is considered.
            logger.info(
                f"Could not determine cutout bounds for input data {idata + 1} "
                f"({exc}), reprojecting to the full output instead"
            )
            edges_out = np.array([np.nan])

        # Determine the cutout parameters

        # In some cases, images might not have valid coordinates in the corners,
        # such as all-sky images or full solar disk views. In this case we skip
        # this step and just use the full output WCS for reprojection.

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
            logger.info("Skipping reprojection as no predicted overlap with final mosaic header")
            continue

        slice_out = tuple([slice(imin, imax) for (imin, imax) in bounds[n_wcs_out_extra:]])

        if isinstance(wcs_out, WCS):
            wcs_out_indiv = wcs_out[slice_out]
        else:
            wcs_out_indiv = SlicedLowLevelWCS(wcs_out.low_level_wcs, slice_out)

        shape_out_indiv = tuple([imax - imin for (imin, imax) in bounds])

        global_block_size = block_sizes[idata]

        # If the block size matches the output shape along the reprojected
        # dimensions, we need to shrink those entries to match this cutout's
        # size (keeping the broadcasted entries) so that the per-image
        # reprojection parallelizes over the broadcasted dimensions.
        if (
            global_block_size
            and not isinstance(global_block_size, str)
            and tuple(global_block_size[-n_dim_reproject:]) == tuple(shape_out[-n_dim_reproject:])
        ):
            block_size = tuple(global_block_size[:n_broadcasted]) + tuple(
                shape_out_indiv[n_broadcasted:]
            )
        else:
            block_size = global_block_size

        yield _InputCutout(
            array_in=array_in,
            wcs_in=wcs_in,
            weights_in=weights_in,
            weights_wcs=weights_wcs,
            bounds=bounds,
            wcs_out_indiv=wcs_out_indiv,
            shape_out_indiv=shape_out_indiv,
            block_size=block_size,
        )


def _coadd_dask(
    cutouts,
    *,
    reproject_function,
    combine_function,
    shape_out,
    target_chunks,
    blank_pixel_value,
    hdu_in,
    reproject_kwargs,
):
    # The return_type='dask' co-addition: reproject each cutout lazily, pad it
    # onto the output grid, and combine all the images along a stacking axis,
    # matching the return_type='numpy' path's _combine_array_into_output
    # semantics exactly. The result is returned uncomputed so the whole graph
    # (reprojections and co-addition) is evaluated in one computation by the
    # caller.

    logger = getLogger(__name__)

    dask_arrays = []
    dask_footprints = []

    for cutout in cutouts:
        # Reproject this image (and its weights) lazily, mirroring the return_type='numpy'
        # per-image handling: NaNs are masked out of the array and footprint,
        # any weights are folded into the footprint, and both the array and
        # footprint are padded back onto the full output grid (the uncovered
        # region gets a zero footprint so it drops out of the combine). The
        # footprint is kept so that the combine below can weight by it exactly
        # as the return_type='numpy' path does. Nothing is computed here.
        logger.info(
            f"Calling {reproject_function.__name__} lazily with "
            f"shape_out={cutout.shape_out_indiv} (return_type='dask')"
        )
        # A dask return requires the reprojection to run in blocks, so fall
        # back to automatic block sizes if none was given for this image.
        dask_block_size = cutout.block_size if cutout.block_size is not None else "auto"
        array, footprint = reproject_function(
            (cutout.array_in, cutout.wcs_in),
            output_projection=cutout.wcs_out_indiv,
            shape_out=cutout.shape_out_indiv,
            hdu_in=hdu_in,
            return_footprint=True,
            return_type="dask",
            block_size=dask_block_size,
            **reproject_kwargs,
        )

        if cutout.weights_in is not None:
            weights = reproject_function(
                (cutout.weights_in, cutout.weights_wcs),
                output_projection=cutout.wcs_out_indiv,
                shape_out=cutout.shape_out_indiv,
                hdu_in=hdu_in,
                return_footprint=False,
                return_type="dask",
                block_size=dask_block_size,
                **reproject_kwargs,
            )
        else:
            weights = None

        reset = da.isnan(array)
        if weights is not None:
            reset = reset | da.isnan(weights)
        array = da.where(reset, 0.0, array)
        footprint = da.where(reset, 0.0, footprint)
        if weights is not None:
            footprint = footprint * da.where(reset, 0.0, weights)

        dask_arrays.append(pad_dask_array_to_grid(array, cutout.bounds, shape_out, target_chunks))
        dask_footprints.append(
            pad_dask_array_to_grid(footprint, cutout.bounds, shape_out, target_chunks)
        )

    if not dask_arrays:
        # No image is predicted to overlap the output; return the same blank
        # mosaic and zero footprint as the return_type='numpy' path, lazily.
        output_array = da.full(tuple(shape_out), float(blank_pixel_value), chunks=target_chunks)
        output_footprint = da.zeros(tuple(shape_out), chunks=target_chunks)
        return output_array, output_footprint

    # Each padded image has chunks aligned to the output chunking, with at most
    # two extra chunk boundaries per dimension at the cutout edges. Rechunking to
    # the common output chunking here therefore only merges those, and stacking
    # identically-chunked arrays keeps the stack and reduction below small
    # (stacking mismatched chunk grids would make dask unify them into an
    # ever-finer grid, with the task count growing super-linearly with the
    # number of images).
    dask_arrays = [array.rechunk(target_chunks) for array in dask_arrays]
    dask_footprints = [footprint.rechunk(target_chunks) for footprint in dask_footprints]
    stacked_array = da.stack(dask_arrays)
    stacked_footprint = da.stack(dask_footprints)

    if combine_function in ("mean", "sum"):
        # Footprint-weighted sum: output = sum(array * footprint), footprint =
        # sum(footprint), and for the mean divide the two (as the
        # return_type='numpy' path does, including contributions where the
        # footprint is negative, which can happen when reprojecting weight maps
        # with interpolation orders that overshoot). The numerator is zero
        # wherever the footprint is zero, so for the mean divide by a footprint
        # that has its zeros replaced by one to avoid a lazily-evaluated 0/0
        # (which would warn at compute time).
        output_array = (stacked_array * stacked_footprint).sum(axis=0)
        output_footprint = stacked_footprint.sum(axis=0)
        if combine_function == "mean":
            output_array = output_array / da.where(output_footprint == 0, 1.0, output_footprint)
    elif combine_function == "median":
        covered = stacked_footprint > 0
        # Unweighted median of the covered images. This is only available for the
        # dask path (the return_type='numpy' path cannot compute a median on the fly).
        masked = da.where(covered, stacked_array, np.nan)
        # Pixels covered by no image at all would be all-NaN slices, which would
        # make nanmedian warn at compute time; give them a value of zero since
        # they are set to blank_pixel_value below anyway.
        masked = da.where(covered.any(axis=0)[np.newaxis], masked, 0.0)
        # The median needs the whole stacking axis at once, so collapse it to a
        # single chunk and let dask re-split the other axes so that the total
        # chunk size stays bounded instead of growing with the number of images.
        masked = masked.rechunk({0: -1, **{iaxis: "auto" for iaxis in range(1, masked.ndim)}})
        output_array = da.nanmedian(masked, axis=0).rechunk(target_chunks)
        output_footprint = stacked_footprint.sum(axis=0)
    else:
        # first/last/min/max select one image per pixel; we find its index along
        # the stacking axis and take both the value and the footprint from there.
        covered = stacked_footprint > 0
        n_images = stacked_array.shape[0]
        axis_index = da.arange(n_images).reshape((n_images,) + (1,) * (stacked_array.ndim - 1))
        if combine_function == "first":
            selected = da.where(covered, axis_index, n_images).min(axis=0)
        elif combine_function == "last":
            selected = da.where(covered, axis_index, -1).max(axis=0)
        elif combine_function == "min":
            selected = da.where(covered, stacked_array, np.inf).argmin(axis=0)
        elif combine_function == "max":
            selected = da.where(covered, stacked_array, -np.inf).argmax(axis=0)
        else:
            raise ValueError(f"Unexpected combine_function: {combine_function}")
        # Pick the value and footprint from the selected image with a one-hot
        # mask along the stacking axis (dask has no take_along_axis). For pixels
        # covered by no image, the selected index either matches no image
        # (first/last) or picks an image whose footprint there is zero (min/max),
        # so the sums give a zero footprint and the pixel is blanked below.
        onehot = axis_index == selected[np.newaxis]
        output_array = da.where(onehot, stacked_array, 0.0).sum(axis=0)
        output_footprint = da.where(onehot, stacked_footprint, 0.0).sum(axis=0)

    # Match the return_type='numpy' path's final step: set pixels with no coverage
    # to the blank value (the return_type='numpy' loop does this at the end where
    # output_footprint == 0, keeping pixels with a negative summed footprint).
    output_array = da.where(output_footprint == 0, blank_pixel_value, output_array)
    return output_array, output_footprint


def _coadd_numpy(
    cutouts,
    *,
    reproject_function,
    combine_function,
    match_background,
    background_reference,
    output_array,
    output_footprint,
    intermediate_memmap,
    blank_pixel_value,
    hdu_in,
    reproject_kwargs,
):
    # The return_type='numpy' co-addition: reproject each cutout and combine it
    # into the output arrays, either on the fly or, when the backgrounds need to
    # be matched, after all the images have been reprojected.

    logger = getLogger(__name__)

    # Define 'on-the-fly' mode: in the case where we don't need to match the
    # backgrounds, we don't have to keep track of the intermediate arrays and
    # can just modify the output array on-the-fly
    on_the_fly = not match_background

    on_the_fly_prefix = "Using" if on_the_fly else "Not using"
    logger.info(
        f"{on_the_fly_prefix} on-the-fly mode for adding individual reprojected images to output array"
    )

    if not on_the_fly:
        arrays = []

    if combine_function == "min":
        output_array[...] = np.inf
    elif combine_function == "max":
        output_array[...] = -np.inf

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=IS_WIN) as local_tmp_dir:

        for cutout in cutouts:

            # TODO: optimize handling of weights by making reprojection functions
            # able to handle weights, and make the footprint become the combined
            # footprint + weight map

            if intermediate_memmap:

                array_path = os.path.join(local_tmp_dir, f"array_{uuid.uuid4()}.np")

                logger.info(
                    f"Creating memory-mapped array with shape {cutout.shape_out_indiv} at {array_path}"
                )

                array = np.memmap(
                    array_path,
                    shape=cutout.shape_out_indiv,
                    mode="w+",
                    dtype=cutout.array_in.dtype,
                )

                footprint_path = os.path.join(local_tmp_dir, f"footprint_{uuid.uuid4()}.np")

                logger.info(
                    f"Creating memory-mapped footprint with shape {cutout.shape_out_indiv} at {footprint_path}"
                )

                footprint = np.memmap(
                    footprint_path,
                    shape=cutout.shape_out_indiv,
                    mode="w+",
                    dtype=float,
                )

            else:

                array = footprint = None

            logger.info(
                f"Calling {reproject_function.__name__} with shape_out={cutout.shape_out_indiv}"
            )

            array, footprint = reproject_function(
                (cutout.array_in, cutout.wcs_in),
                output_projection=cutout.wcs_out_indiv,
                shape_out=cutout.shape_out_indiv,
                hdu_in=hdu_in,
                output_array=array,
                output_footprint=footprint,
                block_size=cutout.block_size,
                **reproject_kwargs,
            )

            if cutout.weights_in is not None:

                if intermediate_memmap:

                    weights_path = os.path.join(local_tmp_dir, f"weights_{uuid.uuid4()}.np")

                    logger.info(
                        f"Creating memory-mapped weights with shape {cutout.shape_out_indiv} at {weights_path}"
                    )

                    weights = np.memmap(
                        weights_path,
                        shape=cutout.shape_out_indiv,
                        mode="w+",
                        dtype=float,
                    )

                else:

                    weights = None

                logger.info(
                    f"Calling {reproject_function.__name__} with shape_out={cutout.shape_out_indiv} for weights"
                )

                weights = reproject_function(
                    (cutout.weights_in, cutout.weights_wcs),
                    output_projection=cutout.wcs_out_indiv,
                    shape_out=cutout.shape_out_indiv,
                    hdu_in=hdu_in,
                    output_array=weights,
                    return_footprint=False,
                    **reproject_kwargs,
                )

            # For the purposes of mosaicking, we mask out NaN values from the array
            # and set the footprint to 0 at these locations. We do this in chunks
            # to avoid excessive memory usage.
            for chunk in iterate_chunks(array.shape, max_chunk_size=DEFAULT_MAX_CHUNK_SIZE):

                # Determine location of NaNs
                reset = np.isnan(array[chunk])
                if cutout.weights_in is not None:
                    reset |= np.isnan(weights[chunk])

                # Mask them in-place in the arrays
                array[chunk][reset] = 0.0
                footprint[chunk][reset] = 0.0

                # Combine weights and footprint
                if cutout.weights_in is not None:
                    weights[chunk][reset] = 0.0
                    footprint[chunk] *= weights[chunk]

            if cutout.weights_in is not None and intermediate_memmap:
                # Remove the reference to the memmap before trying to remove the file itself
                logger.info("Removing memory-mapped weight array")
                weights = None
                _safe_remove(weights_path)

            array = ReprojectedArraySubset(array, footprint, cutout.bounds)

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

        if combine_function == "mean":
            logger.info("Handle normalization of output array")
            with np.errstate(invalid="ignore"):
                output_array /= output_footprint

        # Avoid keeping any references to the memory-mapped arrays so that the
        # files get cleaned up once we exit the context manager.
        if intermediate_memmap:
            array = None
            footprint = None
            arrays = []

    # We need to avoid potentially large memory allocation from output == 0 so
    # we operate in chunks.
    logger.info(f"Resetting invalid pixels to {blank_pixel_value}")
    for chunk in iterate_chunks(output_array.shape, max_chunk_size=DEFAULT_MAX_CHUNK_SIZE):
        output_array[chunk][output_footprint[chunk] == 0] = blank_pixel_value

    return output_array, output_footprint


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
    combine_function : { 'mean', 'sum', 'first', 'last', 'min', 'max', 'median' }
        The type of function to use for combining the values into the final
        image. For 'first' and 'last', respectively, the reprojected images are
        simply overlaid on top of each other. With respect to the order of the
        input images in ``input_data``, either the first or the last image to
        cover a region of overlap determines the output data for that region.
        'median' is only available with ``return_type='dask'``.
    match_background : bool
        Whether to match the backgrounds of the images. Only supported with
        ``return_type='numpy'``.
    background_reference : `None` or `int`
        If `None`, the background matching will make it so that the average of
        the corrections for all images is zero. If an integer, this specifies
        the index of the image to use as a reference.
    output_array : array or None
        The final output array.  Specify this if you already have an
        appropriately-shaped array to store the data in.  Must match shape
        specified with ``shape_out`` or derived from the output
        projection. Can only be specified with ``return_type='numpy'``.
    output_footprint : array or None
        The final output footprint array.  Specify this if you already have an
        appropriately-shaped array to store the data in.  Must match shape
        specified with ``shape_out`` or derived from the output projection.
        Can only be specified with ``return_type='numpy'``.
    block_sizes : list of tuples or None
        The block size to use for each dataset.  Could also be a single tuple
        if you want the same block size for all data sets. With
        ``return_type='dask'``, a single common block size is also used as the
        chunking of the returned dask arrays.
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
        reprojected data. Only supported with ``return_type='numpy'``.
    return_type : {None, 'numpy', 'dask'}, optional
        If ``'dask'``, reproject each image lazily (using ``return_type='dask'``
        on the reprojection function), pad each onto the output grid, and combine
        them along a stacking axis, returning the resulting **uncomputed** dask
        arrays so the whole co-addition is computed lazily in one go. The
        combination matches the ``return_type='numpy'`` path exactly
        (footprint-weighted for 'mean' and 'sum', footprint-aware selection for
        'first', 'last', 'min' and 'max'), and ``input_weights`` are supported. A
        ``combine_function`` of 'median' is additionally available here (as an
        unweighted median), which the ``return_type='numpy'`` path cannot compute.
        ``match_background``, ``output_array``, ``output_footprint`` and
        ``intermediate_memmap`` are not supported, since the result is an
        uncomputed graph rather than arrays filled in place. The default (`None`)
        is equivalent to ``'numpy'``.

    **kwargs
        Keyword arguments to be passed to the reprojection function.

    Returns
    -------
    array : `~numpy.ndarray` or `~dask.array.Array`
        The co-added array. This is an uncomputed dask array when
        ``return_type='dask'``.
    footprint : `~numpy.ndarray` or `~dask.array.Array`
        Footprint of the co-added array. Values of 0 indicate no coverage or
        valid values in the input image, while values of 1 indicate valid
        values. This is an uncomputed dask array when ``return_type='dask'``.
    """

    # TODO: add support for saving intermediate files to disk to avoid blowing
    # up memory usage. We could probably still have references to array
    # objects, but we'd just make sure these were memory mapped

    logger = getLogger(__name__)

    # Validate inputs

    # Validate return_type up front: a typo such as 'Dask' would otherwise
    # silently select the return_type='numpy' co-addition.
    if return_type is None:
        return_type = "numpy"
    if return_type not in ("numpy", "dask"):
        raise ValueError("return_type should be set to 'numpy' or 'dask'")

    # 'median' is only available for the deferred dask path (return_type='dask'); the
    # return_type='numpy' path cannot compute a median on the fly.
    allowed_combine = ("mean", "sum", "first", "last", "min", "max")
    if return_type == "dask":
        allowed_combine = allowed_combine + ("median",)
    if combine_function not in allowed_combine:
        raise ValueError(f"combine_function should be one of {'/'.join(allowed_combine)}")

    if reproject_function is None:
        raise ValueError(
            "reprojection function should be specified with the reproject_function argument"
        )

    if "block_size" in kwargs and kwargs["block_size"] is not None:
        if block_sizes is not None:
            raise ValueError("Cannot specify block_sizes= and block_size= at the same time")
        block_sizes = kwargs.pop("block_size")

    # block_sizes may be a single block size to use for every dataset (a tuple
    # of values or 'auto') or one block size per dataset (a list of lists or
    # tuples). Normalize it once here to one entry per dataset, keeping track
    # of the single common block size (if there is one), which is also used as
    # the output chunking for the deferred return_type='dask' co-addition.
    common_block_size = None
    if block_sizes is None:
        block_sizes = [None] * len(input_data)
    elif isinstance(block_sizes, str) or not any(
        isinstance(entry, (list, tuple)) for entry in block_sizes
    ):
        common_block_size = block_sizes if isinstance(block_sizes, str) else tuple(block_sizes)
        block_sizes = [common_block_size] * len(input_data)
    elif len(block_sizes) != len(input_data):
        raise ValueError(
            f"block_sizes should be a single block size or one per dataset "
            f"({len(input_data)}), got {len(block_sizes)}"
        )

    # non_reprojected_dims is used below to size the cutouts correctly and is
    # also forwarded to reproject_function for each individual reprojection.
    # Validate it here too since reproject_function performs the same check but
    # is never called if no input is predicted to overlap the output, in which
    # case an invalid value would otherwise silently produce a blank mosaic.
    if non_reprojected_dims is not None:
        if non_reprojected_dims != tuple(range(len(non_reprojected_dims))):
            raise ValueError(
                "non_reprojected_dims should be a tuple with values increasing sequentially from zero"
            )
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
        if output_array is not None or output_footprint is not None:
            raise ValueError(
                "Cannot use return_type='dask' together with output_array or output_footprint"
            )
        if intermediate_memmap:
            raise ValueError("Cannot use return_type='dask' together with intermediate_memmap")
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

    if coadd_with_dask:
        # The deferred co-addition below stacks all the lazily-reprojected images,
        # which requires a single common chunking. When the user gave a single
        # common block size, use it as the output chunking so they stay in control
        # (a per-dataset list of block sizes cannot map onto one output chunking,
        # so it falls back to the default). Otherwise default to splitting along
        # the non-reprojected (leading) dimensions, so each output chunk is a
        # single plane rather than the whole non-reprojected extent -- that keeps
        # memory bounded to a plane per image and lets the reprojection and
        # co-addition stream plane by plane instead of reprojecting every image
        # in full before combining.
        block_size = None
        if common_block_size is not None and not isinstance(common_block_size, str):
            block_size = common_block_size

        if block_size is None:
            chunk_spec = (1,) * n_broadcasted + ("auto",) * n_dim_reproject
        elif len(block_size) == n_dim_reproject:
            # A block size given only for the reprojected dimensions; take one
            # plane at a time along the non-reprojected ones.
            chunk_spec = (1,) * n_broadcasted + block_size
        else:
            chunk_spec = block_size
        target_chunks = da.core.normalize_chunks(chunk_spec, shape=tuple(shape_out), dtype=float)

    cutouts = _input_cutout_iterator(
        input_data,
        input_weights,
        hdu_in,
        hdu_weights,
        wcs_out,
        shape_out,
        block_sizes,
        n_broadcasted,
        n_dim_reproject,
        n_wcs_out_extra,
        progress_bar,
    )

    if coadd_with_dask:
        return _coadd_dask(
            cutouts,
            reproject_function=reproject_function,
            combine_function=combine_function,
            shape_out=shape_out,
            target_chunks=target_chunks,
            blank_pixel_value=blank_pixel_value,
            hdu_in=hdu_in,
            reproject_kwargs=kwargs,
        )
    else:
        return _coadd_numpy(
            cutouts,
            reproject_function=reproject_function,
            combine_function=combine_function,
            match_background=match_background,
            background_reference=background_reference,
            output_array=output_array,
            output_footprint=output_footprint,
            intermediate_memmap=intermediate_memmap,
            blank_pixel_value=blank_pixel_value,
            hdu_in=hdu_in,
            reproject_kwargs=kwargs,
        )

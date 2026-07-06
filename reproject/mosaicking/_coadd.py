# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os
import sys
import tempfile
import uuid
import warnings
from collections import namedtuple
from logging import getLogger

import dask
import dask.array as da
import numpy as np
from astropy.wcs import WCS
from astropy.wcs.wcsapi import SlicedLowLevelWCS

from .._array_utils import iterate_chunks
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


def _combine_piece(combine_function, array, footprint, output_array, output_footprint):
    # Combine one piece of a reprojected image into the output array and
    # footprint, where output_array and output_footprint are views covering
    # the same region as the piece. This is the per-pixel logic shared between
    # the return_type='numpy' and return_type='dask' co-addition paths; in
    # both, the images are applied in input order, which is what gives 'first'
    # and 'last' their sequential meaning.
    if combine_function in ("mean", "sum"):
        output_footprint += footprint
        output_array += array * footprint
    elif combine_function in ("first", "last", "min", "max"):
        if combine_function == "first":
            mask = (footprint > 0) & (output_footprint == 0)
        elif combine_function == "last":
            mask = footprint > 0
        elif combine_function == "min":
            mask = (footprint > 0) & (array < output_array)
        elif combine_function == "max":
            mask = (footprint > 0) & (array > output_array)

        # Update only the selected pixels in place, which avoids allocating
        # and rewriting the whole chunk as np.where would.
        np.copyto(output_footprint, footprint, where=mask)
        np.copyto(output_array, array, where=mask)
    else:
        raise ValueError(f"Unexpected combine_function: {combine_function}")


def _combine_array_into_output(combine_function, array, output_array, output_footprint):
    for chunk in array.as_chunks():
        # Values outside of the footprint are set to NaN by default
        # but we set these to 0 here to avoid NaNs in the means/sums.
        if combine_function in ("mean", "sum"):
            chunk.array[chunk.footprint == 0] = 0.0
        _combine_piece(
            combine_function,
            chunk.array,
            chunk.footprint,
            output_array[chunk.view_in_original_array],
            output_footprint[chunk.view_in_original_array],
        )


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
        # size (keeping any leading entries, which the block size may omit
        # entirely if it only covers the reprojected dimensions) so that the
        # per-image reprojection parallelizes over the broadcasted dimensions.
        if (
            global_block_size
            and not isinstance(global_block_size, str)
            and tuple(global_block_size[-n_dim_reproject:]) == tuple(shape_out[-n_dim_reproject:])
        ):
            block_size = tuple(global_block_size[:-n_dim_reproject]) + tuple(
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


def _combine_tile_pieces(
    *pieces, combine_function, blank_pixel_value, dests, trailing_shape, block_info=None
):
    # Combine the pieces of the reprojected images that overlap one chunk of the
    # output, using the same per-pixel logic as the return_type='numpy' path
    # (shared through _combine_piece). pieces holds an array block and a footprint
    # block for each overlapping image, each covering dests[i] within the
    # chunk; when no image overlaps (dests is empty), a single zeros template
    # just provides the leading shape and the chunk comes out blank.
    if dests:
        arrays = pieces[0::2]
        footprints = pieces[1::2]
    else:
        arrays = footprints = ()
    lead_shape = pieces[0].shape[: pieces[0].ndim - len(trailing_shape)]
    shape = lead_shape + tuple(trailing_shape)

    output_array = np.zeros(shape)
    output_footprint = np.zeros(shape)
    if combine_function == "min":
        output_array[...] = np.inf
    elif combine_function == "max":
        output_array[...] = -np.inf

    if combine_function == "median":
        # Unweighted median of the values covering each pixel. Pixels covered
        # by no image would be all-NaN slices, which make nanmedian warn; give
        # them a dummy value of zero since they are blanked below anyway.
        values = np.full((max(len(dests), 1),) + shape, np.nan)
        for index, (array, footprint, dest) in enumerate(
            zip(arrays, footprints, dests, strict=True)
        ):
            region = (Ellipsis,) + dest
            values[index][region] = np.where(footprint > 0, array, np.nan)
            output_footprint[region] += footprint
        values[0][np.isnan(values).all(axis=0)] = 0.0
        output_array = np.nanmedian(values, axis=0)
    else:
        # For the mean, divide the footprint-weighted sum by the summed
        # footprint (as the return_type='numpy' path does, including
        # contributions where the footprint is negative, which can happen when
        # reprojecting weight maps with interpolation orders that overshoot);
        # zeros in the divisor are replaced by one to avoid a 0/0 warning,
        # since those pixels are set to the blank value below anyway.
        for array, footprint, dest in zip(arrays, footprints, dests, strict=True):
            region = (Ellipsis,) + dest
            _combine_piece(
                combine_function, array, footprint, output_array[region], output_footprint[region]
            )
        if combine_function == "mean":
            output_array /= np.where(output_footprint == 0, 1, output_footprint)

    # Match the return_type='numpy' path's final step: set pixels with no
    # coverage to the blank value (only where the summed footprint is exactly
    # zero, keeping pixels with a negative summed footprint).
    output_array = np.where(output_footprint == 0, blank_pixel_value, output_array)
    return np.stack([output_array, output_footprint])


def _coadd_dask(
    cutouts,
    *,
    reproject_function,
    combine_function,
    shape_out,
    target_chunks,
    n_dim_reproject,
    blank_pixel_value,
    hdu_in,
    reproject_kwargs,
):
    # The return_type='dask' co-addition: reproject each cutout lazily, then
    # assemble each chunk of the output from only the images that overlap it,
    # matching the return_type='numpy' path's _combine_array_into_output
    # semantics exactly. The result is returned uncomputed so the whole graph
    # (reprojections and co-addition) is evaluated in one computation by the
    # caller.

    logger = getLogger(__name__)

    tiles = []

    # Dask identifies arrays by their name, so two different input arrays that
    # share a name (a bug seen in the wild for arrays built with a hard-coded
    # graph layer name) are silently treated as the same array once everything
    # is combined into a single graph, with one input's data used for all of
    # them. Passing the same array object several times is fine.
    seen_dask_arrays = {}

    for cutout in cutouts:
        if isinstance(cutout.array_in, da.core.Array):
            previous = seen_dask_arrays.setdefault(cutout.array_in.name, cutout.array_in)
            if previous is not cutout.array_in:
                warnings.warn(
                    f"Two different input dask arrays share the name "
                    f"{cutout.array_in.name!r}, so dask will treat them as the "
                    "same array in the combined co-addition graph and the "
                    "resulting mosaic will likely use one input's data for "
                    "both. Make sure each input dask array has a unique name.",
                    UserWarning,
                    stacklevel=2,
                )
        # Reproject this image (and its weights) lazily, mirroring the return_type='numpy'
        # per-image handling: NaNs are masked out of the array and footprint,
        # and any weights are folded into the footprint, which is kept so that
        # the combine below can weight by it exactly as the return_type='numpy'
        # path does. Nothing is computed here.
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

        tiles.append((array, footprint, cutout.bounds))

    if not tiles:
        # No image is predicted to overlap the output; return the same blank
        # mosaic and zero footprint as the return_type='numpy' path, lazily.
        output_array = da.full(tuple(shape_out), float(blank_pixel_value), chunks=target_chunks)
        output_footprint = da.zeros(tuple(shape_out), chunks=target_chunks)
        return output_array, output_footprint

    # Assemble each chunk of the output from only the images that overlap it:
    # for every column of output chunks along the reprojected dimensions,
    # slice the overlapping region out of each reprojected image and combine
    # the pieces chunk by chunk in _combine_tile_pieces. Compared to padding
    # every image onto the full output grid and reducing along a stacking
    # axis, no zero chunks are ever materialized and each output chunk depends
    # only on the images that actually cover it, so both the graph size and
    # the amount of data resident during the computation are proportional to
    # the true overlap rather than to n_images x n_chunks.
    n_lead = len(shape_out) - n_dim_reproject
    lead_chunks = target_chunks[:n_lead]
    trailing_chunks = target_chunks[n_lead:]
    edges = [np.concatenate([[0], np.cumsum(chunks)]) for chunks in trailing_chunks]

    columns = np.empty(tuple(len(chunks) for chunks in trailing_chunks), dtype=object)
    for index in np.ndindex(columns.shape):
        extents = [(int(edges[idim][i]), int(edges[idim][i + 1])) for idim, i in enumerate(index)]
        pieces = []
        dests = []
        for array, footprint, bounds in tiles:
            trailing_bounds = bounds[n_lead:]
            overlap = [
                (max(lo, imin), min(hi, imax))
                for (lo, hi), (imin, imax) in zip(extents, trailing_bounds, strict=True)
            ]
            if any(hi <= lo for lo, hi in overlap):
                continue
            # Slice of the image (in cutout coordinates) that falls inside this
            # column, kept as one chunk along the reprojected dimensions and
            # matching the output chunking along the non-reprojected ones.
            local = (slice(None),) * n_lead + tuple(
                slice(lo - imin, hi - imin)
                for (lo, hi), (imin, _) in zip(overlap, trailing_bounds, strict=True)
            )
            rechunk_spec = {idim: lead_chunks[idim] for idim in range(n_lead)}
            rechunk_spec.update({n_lead + idim: -1 for idim in range(n_dim_reproject)})
            pieces.append(array[local].rechunk(rechunk_spec))
            pieces.append(footprint[local].rechunk(rechunk_spec))
            # Where the piece lands within the column's chunks.
            dests.append(
                tuple(
                    slice(lo - start, hi - start)
                    for (lo, hi), (start, _) in zip(overlap, extents, strict=True)
                )
            )
        chunk_shape = tuple(hi - lo for lo, hi in extents)
        if not pieces:
            # No image overlaps this column; a zeros template just provides the
            # block structure so the chunks come out blank.
            pieces = [
                da.zeros(
                    tuple(shape_out[:n_lead]) + chunk_shape,
                    chunks=lead_chunks + tuple((size,) for size in chunk_shape),
                )
            ]
        columns[index] = da.map_blocks(
            _combine_tile_pieces,
            *pieces,
            combine_function=combine_function,
            blank_pixel_value=blank_pixel_value,
            dests=tuple(dests),
            trailing_shape=chunk_shape,
            dtype=float,
            new_axis=0,
            chunks=((2,),) + lead_chunks + tuple((size,) for size in chunk_shape),
            # Without an explicit meta, map_blocks infers it by calling the
            # combine function on placeholder inputs, which allocates two
            # full-size chunks per column at graph construction time (the
            # placeholders are empty but the kwargs above are not)
            meta=np.empty((0,) * (1 + n_lead + n_dim_reproject), dtype=float),
        )

    result = da.block(columns.tolist())
    return result[0], result[1]


def _coadd_zarr(
    cutouts,
    *,
    reproject_function,
    combine_function,
    shape_out,
    target_chunks,
    n_dim_reproject,
    blank_pixel_value,
    hdu_in,
    reproject_kwargs,
    zarr_path,
    zarr_batch_size,
):
    # The return_type='zarr' co-addition: build the same deferred graphs as
    # return_type='dask', but compute them here, batch by batch along the
    # first dimension, into a zarr store on disk. Each batch is built as its
    # own graph from only the images that overlap it, so the memory used by
    # the computation is bounded by one batch regardless of the size of the
    # mosaic (the scheduler never sees tasks from more than one batch), and
    # each batch is written to a disjoint region of the store. Reprojected
    # chunks of images that span a batch boundary are computed in each batch
    # that needs them, trading some recomputation for the bounded memory.

    import zarr  # noqa: PLC0415

    logger = getLogger(__name__)

    # Compute with the scheduler implied by the parallel keyword, following
    # the same semantics as the individual reprojection functions:
    # 'current-scheduler' uses the active scheduler (e.g. dask.distributed),
    # an integer uses that many threads, True uses the default number of
    # threads, and False computes synchronously.
    parallel = reproject_kwargs.get("parallel", False)
    if parallel == "current-scheduler":
        scheduler_config = {}
    elif isinstance(parallel, bool):
        scheduler_config = {"scheduler": "threads"} if parallel else {"scheduler": "synchronous"}
    elif parallel > 0:
        scheduler_config = {"scheduler": "threads", "num_workers": parallel}
    else:
        raise ValueError("The number of processors to use must be strictly positive")

    cutouts = list(cutouts)

    chunk_shape = tuple(chunks[0] for chunks in target_chunks)

    group = zarr.open_group(zarr_path, mode="w-")
    # zarr 2 calls the array creation method create_dataset
    create = group.create_array if hasattr(group, "create_array") else group.create_dataset
    # Batches with no overlapping images are never written, so the fill
    # values provide the same blank mosaic values as the other paths
    zarr_array = create(
        "array",
        shape=tuple(shape_out),
        chunks=chunk_shape,
        dtype=float,
        fill_value=float(blank_pixel_value),
    )
    zarr_footprint = create(
        "footprint",
        shape=tuple(shape_out),
        chunks=chunk_shape,
        dtype=float,
        fill_value=0.0,
    )

    # Batches are groups of output chunks, iterating over the chunk grid in
    # C order, sized so that one batch of the output is around 2 GB by
    # default, since the working memory of the computation scales with the
    # batch size.
    numblocks = tuple(len(chunks) for chunks in target_chunks)
    all_edges = [np.concatenate([[0], np.cumsum(chunks)]).astype(int) for chunks in target_chunks]

    if zarr_batch_size is None:
        chunk_nbytes = float(np.prod(chunk_shape, dtype=float)) * 8
        zarr_batch_size = max(1, int(2 * 1024**3 // max(chunk_nbytes, 1)))

    chunk_indices = list(np.ndindex(numblocks))
    n_batches = int(np.ceil(len(chunk_indices) / zarr_batch_size))
    for ibatch in range(n_batches):
        batch = chunk_indices[ibatch * zarr_batch_size : (ibatch + 1) * zarr_batch_size]

        regions = [
            tuple(
                slice(int(all_edges[axis][index[axis]]), int(all_edges[axis][index[axis] + 1]))
                for axis in range(len(numblocks))
            )
            for index in batch
        ]

        # Keep only the images that overlap at least one chunk in the batch
        batch_cutouts = [
            cutout
            for cutout in cutouts
            if any(
                all(
                    cutout.bounds[axis][1] > region[axis].start
                    and cutout.bounds[axis][0] < region[axis].stop
                    for axis in range(len(numblocks))
                )
                for region in regions
            )
        ]
        logger.info(
            f"Computing batch {ibatch + 1} of {n_batches} ({len(batch)} chunks, "
            f"{len(batch_cutouts)} overlapping images)"
        )
        if not batch_cutouts:
            continue
        array, footprint = _coadd_dask(
            batch_cutouts,
            reproject_function=reproject_function,
            combine_function=combine_function,
            shape_out=shape_out,
            target_chunks=target_chunks,
            n_dim_reproject=n_dim_reproject,
            blank_pixel_value=blank_pixel_value,
            hdu_in=hdu_in,
            reproject_kwargs=reproject_kwargs,
        )
        sources = [array[region] for region in regions] + [footprint[region] for region in regions]
        targets = [zarr_array] * len(regions) + [zarr_footprint] * len(regions)
        with dask.config.set(**scheduler_config):
            da.store(
                sources,
                targets,
                regions=regions + regions,
                lock=False,
                compute=True,
            )

    return (
        da.from_zarr(zarr_path, component="array"),
        da.from_zarr(zarr_path, component="footprint"),
    )


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
                    block_size=cutout.block_size,
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
    zarr_path=None,
    zarr_batch_size=None,
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
    return_type : {None, 'numpy', 'dask', 'zarr'}, optional
        If ``'dask'``, reproject each image lazily (using ``return_type='dask'``
        on the reprojection function) and assemble each chunk of the output
        from the images that overlap it, returning the resulting **uncomputed**
        dask arrays so the whole co-addition is computed lazily in one go. The
        combination matches the ``return_type='numpy'`` path exactly
        (footprint-weighted for 'mean' and 'sum', footprint-aware selection for
        'first', 'last', 'min' and 'max'), and ``input_weights`` are supported. A
        ``combine_function`` of 'median' is additionally available here (as an
        unweighted median), which the ``return_type='numpy'`` path cannot compute.
        ``match_background``, ``output_array``, ``output_footprint`` and
        ``intermediate_memmap`` are not supported, since the result is an
        uncomputed graph rather than arrays filled in place. If ``'zarr'``,
        build the same graphs as ``'dask'`` but compute them here, batch by
        batch along the first dimension, into a zarr store at ``zarr_path``,
        returning dask arrays that read from the store. This bounds the memory
        used by the computation to one batch regardless of the size of the
        mosaic, at the cost of recomputing reprojected chunks of images that
        span a batch boundary, and the restrictions of ``'dask'`` apply. The
        default (`None`) is equivalent to ``'numpy'``.
    zarr_path : str, optional
        The path at which to create the zarr store when
        ``return_type='zarr'``. The path must not already exist. The store is
        created as a group with ``'array'`` and ``'footprint'`` arrays.
    zarr_batch_size : int, optional
        The number of output chunks to compute per batch when
        ``return_type='zarr'``, iterating over the output chunks in C order.
        The default picks a batch size such that one batch of the output is
        around 2 GB. The batches are computed with the scheduler implied by
        the ``parallel`` keyword, with the same semantics as for the
        individual reprojection functions (``'current-scheduler'`` to use the
        active scheduler, an integer for that many threads, `True` for the
        default number of threads and `False` to compute synchronously).

    **kwargs
        Keyword arguments to be passed to the reprojection function.

    Returns
    -------
    array : `~numpy.ndarray` or `~dask.array.Array`
        The co-added array. This is an uncomputed dask array when
        ``return_type='dask'``, and a dask array reading from the computed
        zarr store when ``return_type='zarr'``.
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
    if return_type not in ("numpy", "dask", "zarr"):
        raise ValueError("return_type should be set to 'numpy', 'dask' or 'zarr'")

    if return_type == "zarr":
        if zarr_path is None:
            raise ValueError("zarr_path should be set when using return_type='zarr'")
        if os.path.exists(zarr_path):
            raise ValueError(f"Path {zarr_path} already exists")
    elif zarr_path is not None:
        raise ValueError("zarr_path can only be set when using return_type='zarr'")

    # 'median' is only available for the deferred dask path (return_type='dask'); the
    # return_type='numpy' path cannot compute a median on the fly.
    allowed_combine = ("mean", "sum", "first", "last", "min", "max")
    if return_type in ("dask", "zarr"):
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
    coadd_with_dask = return_type in ("dask", "zarr")

    if coadd_with_dask:
        if match_background:
            raise ValueError(
                f"Cannot use return_type={return_type!r} together with match_background"
            )
        if output_array is not None or output_footprint is not None:
            raise ValueError(
                f"Cannot use return_type={return_type!r} together with output_array or output_footprint"
            )
        if intermediate_memmap:
            raise ValueError(
                f"Cannot use return_type={return_type!r} together with intermediate_memmap"
            )
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

    if return_type == "zarr":
        return _coadd_zarr(
            cutouts,
            reproject_function=reproject_function,
            combine_function=combine_function,
            shape_out=shape_out,
            target_chunks=target_chunks,
            n_dim_reproject=n_dim_reproject,
            blank_pixel_value=blank_pixel_value,
            hdu_in=hdu_in,
            reproject_kwargs=kwargs,
            zarr_path=zarr_path,
            zarr_batch_size=zarr_batch_size,
        )
    elif coadd_with_dask:
        return _coadd_dask(
            cutouts,
            reproject_function=reproject_function,
            combine_function=combine_function,
            shape_out=shape_out,
            target_chunks=target_chunks,
            n_dim_reproject=n_dim_reproject,
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

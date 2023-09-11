# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import uuid
import tempfile
from math import ceil
from itertools import product

import dask
import dask.array as da
import numpy as np
from astropy.wcs import WCS
from astropy.wcs.wcsapi import SlicedLowLevelWCS

from ..utils import parse_input_data, parse_input_weights, parse_output_projection
from .background import determine_offset_matrix, solve_corrections_sgd
from .subset_array import ReprojectedArraySubset

__all__ = ["reproject_and_coadd"]


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
    parallel=False,
    block_size=None,
    return_type="numpy",
    **kwargs,
):
    """
    Given a set of input images, reproject and co-add these to a single
    final image.

    This currently only works with 2-d images with celestial WCS.

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
    combine_function : { 'mean', 'sum', 'median', 'first', 'last', 'min', 'max' }
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
        specified with ``shape_out`` or derived from the output projection.
    output_footprint : array or None
        The final output footprint array.  Specify this if you already have an
        appropriately-shaped array to store the data in.  Must match shape
        specified with ``shape_out`` or derived from the output projection.
    parallel : bool or int
        Flag for parallel implementation. If ``True``, a parallel implementation
        is chosen, the number of processes selected automatically to be equal to
        the number of logical CPUs detected on the machine. If ``False``, a
        serial implementation is chosen. If the flag is a positive integer ``n``
        greater than one, a parallel implementation using ``n`` processes is chosen.
    block_size : tuple, optional
        The block size to use for computing the output. Note that this cannot
        be used with the ``match_background`` option.
    return_type : {'numpy', 'dask'}, optional
        Whether to return numpy or dask arrays - defaults to 'numpy'.
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

    if combine_function not in ("mean", "sum", "median", "first", "last", "min", "max"):
        raise ValueError("combine_function should be one of mean/sum/median/first/last/min/max")

    if reproject_function is None:
        raise ValueError(
            "reprojection function should be specified with the reproject_function argument"
        )

    if block_size is not None:
        if match_background:
            raise ValueError("Cannot specify match_background=True and block_size simultaneously")

        if input_weights is not None:
            raise NotImplementedError("Cannot yet specify input weights when block_size is set")

    # Parse the output projection to avoid having to do it for each

    wcs_out, shape_out = parse_output_projection(output_projection, shape_out=shape_out)

    if block_size is None:
        if parallel:
            raise NotImplementedError("Cannot use parallel= if block_size is not set")

        if len(shape_out) != 2:
            raise ValueError(
                "Only 2-dimensional reprojections are supported when block_size is not set"
            )

    else:
        if not isinstance(block_size, tuple):
            block_size = (block_size,) * len(shape_out)

        # Pad shape_out so that it is a multiple of the block size along each dimension
        shape_out_original = shape_out
        shape_out = tuple(
            [ceil(shape_out[i] / block_size[i]) * block_size[i] for i in range(len(shape_out))]
        )

    if output_array is not None and output_array.shape != shape_out:
        raise ValueError(
            "If you specify an output array, it must have a shape matching "
            f"the output shape {shape_out}"
        )
    if output_footprint is not None and output_footprint.shape != shape_out:
        raise ValueError(
            "If you specify an output footprint array, it must have a shape matching "
            f"the output shape {shape_out}"
        )

    # Start off by reprojecting individual images to the final projection

    arrays = []

    for idata in range(len(input_data)):
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
        # for each tile.

        if len(shape_out) == 2:
            # We strike a balance between transforming only the input-image
            # corners, which is fast but can cause clipping in cases of
            # significant distortion (when the edges of the input image become
            # convex in the output projection), and transforming every edge
            # pixel, which provides a lot of redundant information.
            ny, nx = array_in.shape
            n_per_edge = 11
            xs = np.linspace(-0.5, nx - 0.5, n_per_edge)
            ys = np.linspace(-0.5, ny - 0.5, n_per_edge)
            xs = np.concatenate((xs, np.full(n_per_edge, xs[-1]), xs, np.full(n_per_edge, xs[0])))
            ys = np.concatenate((np.full(n_per_edge, ys[0]), ys, np.full(n_per_edge, ys[-1]), ys))
            pixel_in = xs, ys
        else:
            # We use only the corners of cubes and higher dimension datasets
            pixel_in = list(
                zip(*product([(-0.5, shape_out[::-1][i] - 0.5) for i in range(len(shape_out))]))
            )
            pixel_in = pixel_in[0]  # FIXME

        pixel_out = wcs_out.world_to_pixel(*wcs_in.pixel_to_world(*pixel_in))

        # Determine the cutout parameters

        # In some cases, images might not have valid coordinates in the corners,
        # such as all-sky images or full solar disk views. In this case we skip
        # this step and just use the full output WCS for reprojection.

        if any([np.any(np.isnan(c_out)) for c_out in pixel_out]):
            wcs_out_indiv = wcs_out
            shape_out_indiv = shape_out

        else:
            # Determine indices - note the reverse order compared to pixel

            slices_out = []
            shape_out_indiv = []

            empty = False

            for i, c_out in enumerate(pixel_out[::-1]):
                imin = max(0, int(np.floor(c_out.min() + 0.5)))
                imax = min(shape_out[i], int(np.ceil(c_out.max() + 0.5)))

                if imax < imin:
                    empty = True
                    break

                # If block size is given, round to nearest block size
                if block_size is not None:
                    if imin % block_size[i] != 0:
                        imin = (imin // block_size[i]) * block_size[i]
                    if imax % block_size[i] != 0:
                        imax = (imax // block_size[i] + 1) * block_size[i]

                slices_out.append(slice(imin, imax))
                shape_out_indiv.append(imax - imin)

            if empty:
                continue

            shape_out_indiv = tuple(shape_out_indiv)

        if isinstance(wcs_out, WCS):
            wcs_out_indiv = wcs_out[slices_out]
        else:
            wcs_out_indiv = SlicedLowLevelWCS(wcs_out.low_level_wcs, slices_out)

        # TODO: optimize handling of weights by making reprojection functions
        # able to handle weights, and make the footprint become the combined
        # footprint + weight map

        if block_size is None:
            array, footprint = reproject_function(
                (array_in, wcs_in),
                output_projection=wcs_out_indiv,
                shape_out=shape_out_indiv,
                hdu_in=hdu_in,
                return_type="numpy",
                **kwargs,
            )

            if weights_in is not None:
                weights = reproject_function(
                    (weights_in, wcs_in),
                    output_projection=wcs_out_indiv,
                    shape_out=shape_out_indiv,
                    hdu_in=hdu_in,
                    return_footprint=False,
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

            array = ReprojectedArraySubset(
                array,
                footprint,
                slices_out[1].start,
                slices_out[1].stop,
                slices_out[0].start,
                slices_out[0].stop,
            )

            # TODO: make sure we gracefully handle the case where the
            # output image is empty (due e.g. to no overlap).

        else:
            array = reproject_function(
                (array_in, wcs_in),
                output_projection=wcs_out_indiv,
                shape_out=shape_out_indiv,
                hdu_in=hdu_in,
                return_footprint=False,
                return_type="dask",
                parallel=parallel,
                block_size=block_size,
                **kwargs,
            )

            # Pad the array so that it covers the whole output area
            array = da.pad(
                array,
                [(sl.start, shape_out[i] - sl.stop) for sl in slices_out],
                constant_values=np.nan,
            )

        arrays.append(array)

    if block_size is None:
        # If requested, try and match the backgrounds.
        if match_background and len(arrays) > 1:
            offset_matrix = determine_offset_matrix(arrays)
            corrections = solve_corrections_sgd(offset_matrix)
            if background_reference:
                corrections -= corrections[background_reference]
            for array, correction in zip(arrays, corrections, strict=True):
                array.array -= correction

        # At this point, the images are now ready to be co-added.

        if output_array is None:
            output_array = np.zeros(shape_out)
        if output_footprint is None:
            output_footprint = np.zeros(shape_out)

        if combine_function == "min":
            output_array[...] = np.inf
        elif combine_function == "max":
            output_array[...] = -np.inf

        if combine_function in ("mean", "sum"):
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
                    output_array[output_footprint == 0] = 0

        elif combine_function in ("first", "last", "min", "max"):
            for array in arrays:
                if combine_function == "first":
                    mask = (output_footprint[array.view_in_original_array] == 0) & (array.footprint > 0)
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
                    mask, array.array, output_array[array.view_in_original_array]                )

        elif combine_function == "median":
            # Here we need to operate in chunks since we could otherwise run
            # into memory issues

            raise NotImplementedError("combine_function='median' is not yet implemented")

        if combine_function in ("min", "max"):
            output_array[output_footprint == 0] = 0.0

        return output_array, output_footprint

    else:
        # TODO: make use of the footprints e.g. in the weighting for the mean/sum

        stacked_array = da.stack(arrays)

        if combine_function == "mean":
            result = da.nanmean(stacked_array, axis=0)
        elif combine_function == "sum":
            result = da.nansum(stacked_array, axis=0)
        elif combine_function == "max":
            result = da.nanmax(stacked_array, axis=0)
        elif combine_function == "min":
            result = da.nanmin(stacked_array, axis=0)
        else:
            raise NotImplementedError(
                "combine_function={combine_function} not yet implemented when block_size is set"
            )

        result = result[
            tuple([slice(0, shape_out_original[i]) for i in range(len(shape_out_original))])
        ]

        if return_type == "dask":
            return result, None

        with tempfile.TemporaryDirectory() as tmp_dir:
            if parallel:
                # As discussed in https://github.com/dask/dask/issues/9556, da.store
                # will not work well in multiprocessing mode when the destination is a
                # Numpy array. Instead, in this case we save the dask array to a zarr
                # array on disk which can be done in parallel, and re-load it as a dask
                # array. We can then use da.store in the next step using the
                # 'synchronous' scheduler since that is I/O limited so does not need
                # to be done in parallel.

                if isinstance(parallel, int):
                    if parallel > 0:
                        workers = {"num_workers": parallel}
                    else:
                        raise ValueError(
                            "The number of processors to use must be strictly positive"
                        )
                else:
                    workers = {}

                zarr_path = os.path.join(tmp_dir, f"{uuid.uuid4()}.zarr")

                with dask.config.set(scheduler="processes", **workers):
                    result.to_zarr(zarr_path)
                result = da.from_zarr(zarr_path)

            if output_array is None:
                return result.compute(scheduler="synchronous"), None
            else:
                da.store(
                    result,
                    output_array,
                    compute=True,
                    scheduler="synchronous",
                )
                return output_array, None

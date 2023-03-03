import tempfile
from pathlib import Path
from concurrent import futures

import astropy.nddata
import dask
import dask.array as da
import numpy as np
from astropy.io import fits
from astropy.io.fits import CompImageHDU, HDUList, Header, ImageHDU, PrimaryHDU
from astropy.wcs import WCS
from astropy.wcs.wcsapi import BaseHighLevelWCS, BaseLowLevelWCS, SlicedLowLevelWCS
from astropy.wcs.wcsapi.high_level_wcs_wrapper import HighLevelWCSWrapper
from dask.utils import SerializableLock

__all__ = [
    "parse_input_data",
    "parse_input_shape",
    "parse_input_weights",
    "parse_output_projection",
]


def parse_input_data(input_data, hdu_in=None):
    """
    Parse input data to return a Numpy array and WCS object.
    """

    if isinstance(input_data, (str, Path)):
        with fits.open(input_data) as hdul:
            return parse_input_data(hdul, hdu_in=hdu_in)
    elif isinstance(input_data, HDUList):
        if hdu_in is None:
            if len(input_data) > 1:
                raise ValueError(
                    "More than one HDU is present, please specify "
                    "HDU to use with ``hdu_in=`` option"
                )
            else:
                hdu_in = 0
        return parse_input_data(input_data[hdu_in])
    elif isinstance(input_data, (PrimaryHDU, ImageHDU, CompImageHDU)):
        return input_data.data, WCS(input_data.header)
    elif isinstance(input_data, tuple) and isinstance(input_data[0], np.ndarray):
        if isinstance(input_data[1], Header):
            return input_data[0], WCS(input_data[1])
        else:
            return input_data
    elif isinstance(input_data, BaseLowLevelWCS) and input_data.array_shape is not None:
        return input_data.array_shape, input_data
    elif isinstance(input_data, astropy.nddata.NDDataBase):
        return input_data.data, input_data.wcs
    else:
        raise TypeError(
            "input_data should either be an HDU object or a tuple "
            "of (array, WCS) or (array, Header)"
        )


def parse_input_shape(input_shape, hdu_in=None):
    """
    Parse input shape information to return an array shape tuple and WCS object.
    """

    if isinstance(input_shape, (str, Path)):
        return parse_input_shape(fits.open(input_shape), hdu_in=hdu_in)
    elif isinstance(input_shape, HDUList):
        if hdu_in is None:
            if len(input_shape) > 1:
                raise ValueError(
                    "More than one HDU is present, please specify "
                    "HDU to use with ``hdu_in=`` option"
                )
            else:
                hdu_in = 0
        return parse_input_shape(input_shape[hdu_in])
    elif isinstance(input_shape, (PrimaryHDU, ImageHDU, CompImageHDU)):
        return input_shape.shape, WCS(input_shape.header)
    elif isinstance(input_shape, tuple) and isinstance(input_shape[0], np.ndarray):
        if isinstance(input_shape[1], Header):
            return input_shape[0].shape, WCS(input_shape[1])
        else:
            return input_shape[0].shape, input_shape[1]
    elif isinstance(input_shape, tuple) and isinstance(input_shape[0], tuple):
        if isinstance(input_shape[1], Header):
            return input_shape[0], WCS(input_shape[1])
        else:
            return input_shape
    elif isinstance(input_shape, BaseLowLevelWCS) and input_shape.array_shape is not None:
        return input_shape.array_shape, input_shape
    elif isinstance(input_shape, astropy.nddata.NDDataBase):
        return input_shape.data.shape, input_shape.wcs
    else:
        raise TypeError(
            "input_shape should either be an HDU object or a tuple "
            "of (array-or-shape, WCS) or (array-or-shape, Header)"
        )


def parse_input_weights(input_weights, hdu_weights=None):
    """
    Parse input weights to return a Numpy array.
    """

    if isinstance(input_weights, str):
        return parse_input_data(fits.open(input_weights), hdu_in=hdu_weights)[0]
    elif isinstance(input_weights, HDUList):
        if hdu_weights is None:
            if len(input_weights) > 1:
                raise ValueError(
                    "More than one HDU is present, please specify "
                    "HDU to use with ``hdu_weights=`` option"
                )
            else:
                hdu_weights = 0
        return parse_input_data(input_weights[hdu_weights])[0]
    elif isinstance(input_weights, (PrimaryHDU, ImageHDU, CompImageHDU)):
        return input_weights.data
    elif isinstance(input_weights, np.ndarray):
        return input_weights
    else:
        raise TypeError("input_weights should either be an HDU object or a Numpy array")


def parse_output_projection(output_projection, shape_in=None, shape_out=None, output_array=None):
    if shape_out is None:
        if output_array is not None:
            shape_out = output_array.shape
    elif shape_out is not None and output_array is not None:
        if shape_out != output_array.shape:
            raise ValueError("shape_out does not match shape of output_array")

    if isinstance(output_projection, Header):
        wcs_out = WCS(output_projection)
        try:
            shape_out = [
                output_projection[f"NAXIS{i + 1}"] for i in range(output_projection["NAXIS"])
            ][::-1]
        except KeyError:
            if shape_out is None:
                raise ValueError(
                    "Need to specify shape since output header "
                    "does not contain complete shape information"
                )
    elif isinstance(output_projection, BaseHighLevelWCS):
        wcs_out = output_projection
        if shape_out is None:
            raise ValueError(
                "Need to specify shape_out when specifying output_projection as WCS object"
            )
    elif isinstance(output_projection, str):
        hdu_list = fits.open(output_projection)
        shape_out = hdu_list[0].data.shape
        header = hdu_list[0].header
        wcs_out = WCS(header)
        hdu_list.close()
    else:
        raise TypeError("output_projection should either be a Header, a WCS object, or a filename")

    if len(shape_out) == 0:
        raise ValueError("The shape of the output image should not be an empty tuple")

    if (
        shape_in is not None
        and len(shape_out) < len(shape_in)
        and len(shape_out) == wcs_out.low_level_wcs.pixel_n_dim
    ):
        # Add the broadcast dimensions to the output shape, which does not
        # currently have any broadcast dims
        shape_out = (*shape_in[: -len(shape_out)], *shape_out)
    return wcs_out, shape_out


def _reproject_blocked(
    reproject_func,
    array_in,
    wcs_in,
    shape_out,
    wcs_out,
    block_size,
    output_array=None,
    return_footprint=True,
    output_footprint=None,
    parallel=True,
):
    """
    Implementation function that handles reprojecting subsets blocks of pixels
    from an input image and holds metadata about where to reinsert when done.

    Parameters
    ----------
    reproject_func
        One the existing reproject functions implementing a reprojection algorithm
        that that will be used be used to perform reprojection
    array_in
        Data following the same format as expected by underlying reproject_func,
        expected to `~numpy.ndarray` when used from _reproject_blocked()
    wcs_in: `~astropy.wcs.WCS`
        WCS object corresponding to array_in
    shape_out: tuple
        Passed to reproject_func() alongside WCS out to determine image size
    wcs_out: `~astropy.wcs.WCS`
        Output WCS image will be projected to. Normally will correspond to subset of
        total output image when used by _reproject_blocked()
    block_size: tuple
        The size of blocks in terms of output array pixels that each block will handle
        reprojecting. Extending out from (0,0) coords positively, block sizes
        are clamped to output space edges when a block would extend past edge
    output_array : None or `~numpy.ndarray`
        An array in which to store the reprojected data.  This can be any numpy
        array including a memory map, which may be helpful when dealing with
        extremely large files.
    return_footprint : bool
        Whether to return the footprint in addition to the output array.
    output_footprint : None or `~numpy.ndarray`
        An array in which to store the footprint of reprojected data.  This can be
        any numpy array including a memory map, which may be helpful when dealing with
        extremely large files.
    parallel : bool or int
        Flag for parallel implementation. If ``True``, a parallel implementation
        is chosen, the number of processes selected automatically to be equal to
        the number of logical CPUs detected on the machine. If ``False``, a
        serial implementation is chosen. If the flag is a positive integer ``n``
        greater than one, a parallel implementation using ``n`` processes is chosen.
    """

    if output_array is None:
        output_array = np.zeros(shape_out, dtype=float)
    if output_footprint is None and return_footprint:
        output_footprint = np.zeros(shape_out, dtype=float)

    if block_size is not None and len(block_size) < len(shape_out):
        block_size = [-1] * (len(shape_out) - len(block_size)) + list(block_size)

    shape_in = array_in.shape

    # When in parallel mode, we want to make sure we avoid having to copy the
    # input array to all processes for each chunk, so instead we write out
    # the input array to a Numpy memory map and load it in inside each process
    # as a memory-mapped array. We need to be careful how this gets passed to
    # reproject_single_block so we pass a variable that can be either a string
    # or the array itself (for synchronous mode).
    if parallel:
        array_in_or_path = tempfile.mktemp()
        array_in_memmapped = np.memmap(
            array_in_or_path, dtype=float, shape=array_in.shape, mode="w+"
        )
        array_in_memmapped[:] = array_in[:]
    else:
        array_in_or_path = array_in

    def reproject_single_block(a, block_info=None):
        if a.ndim == 0 or block_info is None or block_info == []:
            return np.array([a, a])
        slices = [slice(*x) for x in block_info[None]["array-location"][-wcs_out.pixel_n_dim :]]
        wcs_out_sub = HighLevelWCSWrapper(SlicedLowLevelWCS(wcs_out, slices=slices))
        if isinstance(array_in_or_path, str):
            array_in = np.memmap(array_in_or_path, dtype=float, shape=shape_in)
        else:
            array_in = array_in_or_path
        array, footprint = reproject_func(
            array_in, wcs_in, wcs_out_sub, block_info[None]["chunk-shape"][1:]
        )
        return np.array([array, footprint])

    # NOTE: the following array is just used to set up the iteration in map_blocks
    # but isn't actually used otherwise - this is deliberate.
    output_array_dask = da.empty(shape_out, chunks=block_size or "auto")

    result = da.map_blocks(
        reproject_single_block,
        output_array_dask,
        dtype=float,
        new_axis=0,
        chunks=(2,) + output_array_dask.chunksize,
    )

    # Truncate extra elements
    result = result[tuple([slice(None)] + [slice(s) for s in shape_out])]

    if parallel:
        # As discussed in https://github.com/dask/dask/issues/9556, da.store
        # will not work well in multiprocessing mode when the destination is a
        # Numpy array. Instead, in this case we save the dask array to a zarr
        # array on disk which can be done in parallel, and re-load it as a dask
        # array. We can then use da.store in the next step using the
        # 'synchronous' scheduler since that is I/O limited so does not need
        # to be done in parallel.
        filename = tempfile.mktemp()
        if isinstance(parallel, int):
            workers = {"num_workers": parallel}
        else:
            workers = {}
        with dask.config.set(scheduler="processes", **workers):
            result.to_zarr(filename)
        result = da.from_zarr(filename)

    if return_footprint:
        da.store(
            [result[0], result[1]],
            [output_array, output_footprint],
            compute=True,
            scheduler="synchronous",
        )
        return output_array, output_footprint
    else:
        da.store(
            result[0],
            output_array,
            compute=True,
            scheduler="synchronous",
        )
        return output_array

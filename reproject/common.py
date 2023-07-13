import tempfile

import dask
import dask.array as da
import numpy as np
from astropy.wcs.wcsapi import BaseHighLevelWCS, SlicedLowLevelWCS
from astropy.wcs.wcsapi.high_level_wcs_wrapper import HighLevelWCSWrapper

__all__ = ['_reproject_dispatcher']


def _reproject_dispatcher(
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
    Main function that handles either calling the core algorithms directly or
    parallelizing or operating in chunks, using dask.

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

        if isinstance(wcs_out, BaseHighLevelWCS):
            low_level_wcs = SlicedLowLevelWCS(wcs_out.low_level_wcs, slices=slices)
        else:
            low_level_wcs = SlicedLowLevelWCS(wcs_out, slices=slices)
        wcs_out_sub = HighLevelWCSWrapper(low_level_wcs)
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
    if block_size:
        output_array_dask = da.empty(shape_out, chunks=block_size)
    else:
        output_array_dask = da.empty(shape_out).rechunk(block_size_limit=8 * 1024**2)

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

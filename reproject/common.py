import os
import tempfile
import uuid

import dask
import dask.array as da
import numpy as np
from astropy.wcs.wcsapi import BaseHighLevelWCS, SlicedLowLevelWCS
from astropy.wcs.wcsapi.high_level_wcs_wrapper import HighLevelWCSWrapper
from dask import delayed

from .utils import _dask_to_numpy_memmap

__all__ = ["_reproject_dispatcher"]


@delayed(pure=True)
def as_delayed_memmap_path(array, tmp_dir):
    if isinstance(array, da.core.Array):
        array_path, _ = _dask_to_numpy_memmap(array, tmp_dir)
    else:
        array_path = os.path.join(tmp_dir, f"{uuid.uuid4()}.npy")
        array_memmapped = np.memmap(
            array_path,
            dtype=float,
            shape=array.shape,
            mode="w+",
        )
        array_memmapped[:] = array[:]

    return array_path


def _reproject_dispatcher(
    reproject_func,
    *,
    array_in,
    wcs_in,
    shape_out,
    wcs_out,
    block_size=None,
    array_out=None,
    return_footprint=True,
    output_footprint=None,
    parallel=True,
    reproject_func_kwargs=None,
    return_type=None,
):
    """
    Main function that handles either calling the core algorithms directly or
    parallelizing or operating in chunks, using dask.

    Parameters
    ----------
    reproject_func
        One the existing reproject functions implementing a reprojection algorithm
        that that will be used be used to perform reprojection
    array_in : `numpy.ndarray` or `dask.array.Array`
        Numpy or dask input array
    wcs_in: `~astropy.wcs.wcsapi.BaseHighLevelWCS`
        Input data WCS
    shape_out: tuple
        Target shape
    wcs_out: `~astropy.wcs.WCS`
        Target WCS
    block_size: tuple or 'auto', optional
        The size of blocks in terms of output array pixels that each block will handle
        reprojecting. Extending out from (0,0) coords positively, block sizes
        are clamped to output space edges when a block would extend past edge.
        Specifying ``'auto'`` means that reprojection will be done in blocks with
        the block size automatically determined. If ``block_size`` is not
        specified or set to `None`, the reprojection will not be carried out in
        blocks.
    array_out : `~numpy.ndarray`, optional
        An array in which to store the reprojected data.  This can be any numpy
        array including a memory map, which may be helpful when dealing with
        extremely large files.
    return_footprint : bool, optional
        Whether to return the footprint in addition to the output array.
    output_footprint : `~numpy.ndarray`, optional
        An array in which to store the footprint of reprojected data.  This can be
        any numpy array including a memory map, which may be helpful when dealing with
        extremely large files.
    parallel : bool or int, optional
        If `True`, the reprojection is carried out in parallel, and if a
        positive integer, this specifies the number of processes to use.
        The reprojection will be parallelized over output array blocks specified
        by ``block_size`` (if the block size is not set, it will be determined
        automatically).
    reproject_func_kwargs : dict, optional
        Keyword arguments to pass through to ``reproject_func``
    return_type : {'numpy', 'dask'}, optional
        Whether to return numpy or dask arrays - defaults to 'numpy'.
    """

    if return_type is None:
        return_type = "numpy"
    elif return_type not in ("numpy", "dask"):
        raise ValueError("return_type should be set to 'numpy' or 'dask'")

    if reproject_func_kwargs is None:
        reproject_func_kwargs = {}

    # We set up a global temporary directory since this will be used e.g. to
    # store memory mapped Numpy arrays and zarr arrays.

    with tempfile.TemporaryDirectory() as local_tmp_dir:
        if array_out is None:
            array_out = np.zeros(shape_out, dtype=float)
        elif array_out.shape != tuple(shape_out):
            raise ValueError(
                f"Output array shape {array_out.shape} should match " f"shape_out={shape_out}"
            )
        elif (array_out.dtype.kind, array_out.dtype.itemsize) != (
            array_in.dtype.kind,
            array_in.dtype.itemsize,
        ):
            # Note that here we don't care if the endians don't match
            raise ValueError(
                f"Output array dtype {array_out.dtype} should match "
                f"input array dtype ({array_in.dtype})"
            )

        # If neither parallel nor blocked reprojection are requested, we simply
        # call the underlying core reproject function with the full arrays.

        if block_size is None and parallel is False:
            # If a dask array was passed as input, we first convert this to a
            # Numpy memory mapped array

            if return_type != "numpy":
                raise ValueError(
                    "Output cannot be returned as dask arrays "
                    "when parallel=False and no block size has "
                    "been specified"
                )

            if isinstance(array_in, da.core.Array):
                _, array_in = _dask_to_numpy_memmap(array_in, local_tmp_dir)

            try:
                return reproject_func(
                    array_in,
                    wcs_in,
                    wcs_out,
                    shape_out=shape_out,
                    array_out=array_out,
                    return_footprint=return_footprint,
                    output_footprint=output_footprint,
                    **reproject_func_kwargs,
                )
            finally:
                # Clean up reference to numpy memmap
                array_in = None

        if output_footprint is None and return_footprint:
            output_footprint = np.zeros(shape_out, dtype=float)

        shape_in = array_in.shape

        # When in parallel mode, we want to make sure we avoid having to copy the
        # input array to all processes for each chunk, so instead we write out
        # the input array to a Numpy memory map and load it in inside each process
        # as a memory-mapped array. We need to be careful how this gets passed to
        # reproject_single_block so we pass a variable that can be either a string
        # or the array itself (for synchronous mode). If the input array is a dask
        # array we should always write it out to a memmap even in synchronous mode
        # otherwise map_blocks gets confused if it gets two dask arrays and tries
        # to iterate over both.

        if isinstance(array_in, da.core.Array) or parallel:
            # If return_type=='dask',
            if return_type == "dask":
                # We should use a temporary directory that will persist beyond
                # the call to the reproject function.
                tmp_dir = tempfile.mkdtemp()
            else:
                tmp_dir = local_tmp_dir
            array_in_or_path = as_delayed_memmap_path(array_in, tmp_dir)
        else:
            # Here we could set array_in_or_path to array_in_path if it
            # has been set previously, but in synchronous mode it is better to
            # simply pass a reference to the memmap array itself to avoid having
            # to load the memmap inside each reproject_single_block call.
            array_in_or_path = array_in

        def reproject_single_block(a, array_or_path, block_info=None):
            if a.ndim == 0 or block_info is None or block_info == []:
                return np.array([a, a])
            slices = [slice(*x) for x in block_info[None]["array-location"][-wcs_out.pixel_n_dim :]]

            if isinstance(wcs_out, BaseHighLevelWCS):
                low_level_wcs = SlicedLowLevelWCS(wcs_out.low_level_wcs, slices=slices)
            else:
                low_level_wcs = SlicedLowLevelWCS(wcs_out, slices=slices)

            wcs_out_sub = HighLevelWCSWrapper(low_level_wcs)

            if isinstance(array_or_path, str):
                array_in = np.memmap(array_or_path, dtype=float, shape=shape_in)
            else:
                array_in = array_or_path

            if array_or_path is None:
                raise ValueError()

            shape_out = block_info[None]["chunk-shape"][1:]

            array, footprint = reproject_func(
                array_in,
                wcs_in,
                wcs_out_sub,
                shape_out=shape_out,
                array_out=np.zeros(shape_out),
                **reproject_func_kwargs,
            )

            return np.array([array, footprint])

        # NOTE: the following array is just used to set up the iteration in map_blocks
        # but isn't actually used otherwise - this is deliberate.

        if block_size is not None and block_size != "auto":
            if wcs_in.low_level_wcs.pixel_n_dim < len(shape_out):
                if len(block_size) < len(shape_out):
                    block_size = [-1] * (len(shape_out) - len(block_size)) + list(block_size)
                else:
                    for i in range(len(shape_out) - wcs_in.low_level_wcs.pixel_n_dim):
                        if block_size[i] != -1 and block_size[i] != shape_out[i]:
                            raise ValueError(
                                "block shape for extra broadcasted dimensions should cover entire array along those dimensions"
                            )
            array_out_dask = da.empty(shape_out, chunks=block_size)
        else:
            if wcs_in.low_level_wcs.pixel_n_dim < len(shape_out):
                chunks = (-1,) * (len(shape_out) - wcs_in.low_level_wcs.pixel_n_dim)
                chunks += ("auto",) * wcs_in.low_level_wcs.pixel_n_dim
                rechunk_kwargs = {"chunks": chunks}
            else:
                rechunk_kwargs = {}
            array_out_dask = da.empty(shape_out)
            array_out_dask = array_out_dask.rechunk(block_size_limit=8 * 1024**2, **rechunk_kwargs)

        result = da.map_blocks(
            reproject_single_block,
            array_out_dask,
            array_in_or_path,
            dtype=float,
            new_axis=0,
            chunks=(2,) + array_out_dask.chunksize,
        )

        # Ensure that there are no more references to Numpy memmaps
        array_in = None
        array_in_or_path = None

        # Truncate extra elements
        result = result[tuple([slice(None)] + [slice(s) for s in shape_out])]

        if return_type == "dask":
            if return_footprint:
                return result[0], result[1]
            else:
                return result[0]

        # We now convert the dask arrays back to Numpy arrays

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
                    raise ValueError("The number of processors to use must be strictly positive")
            else:
                workers = {}

            zarr_path = os.path.join(local_tmp_dir, f"{uuid.uuid4()}.zarr")

            with dask.config.set(scheduler="processes", **workers):
                result.to_zarr(zarr_path)
            result = da.from_zarr(zarr_path)

        if return_footprint:
            da.store(
                [result[0], result[1]],
                [array_out, output_footprint],
                compute=True,
                scheduler="synchronous",
            )
            return array_out, output_footprint
        else:
            da.store(
                result[0],
                array_out,
                compute=True,
                scheduler="synchronous",
            )
            return array_out

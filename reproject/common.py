import logging
import os
import tempfile
import uuid

import dask
import dask.array as da
import numpy as np
from astropy.wcs import WCS
from astropy.wcs.wcsapi import BaseHighLevelWCS, SlicedLowLevelWCS
from astropy.wcs.wcsapi.high_level_wcs_wrapper import HighLevelWCSWrapper
from dask import delayed

from .utils import _dask_to_numpy_memmap

__all__ = ["_reproject_dispatcher"]


class _ArrayContainer:
    # When we set up as_delayed_memmap_path, if we pass a dask array to it,
    # dask will actually compute the array before we get to the code inside
    # as_delayed_memmap_path, so as a workaround we wrap any array we
    # pass in using _ArrayContainer to make sure dask doesn't try and be smart.
    def __init__(self, array):
        self._array = array


@delayed(pure=True)
def as_delayed_memmap_path(array, tmp_dir):

    # Extract array from _ArrayContainer
    if isinstance(array, _ArrayContainer):
        array = array._array
    else:
        raise TypeError("Expected _ArrayContainer in as_delayed_memmap_path")

    logger = logging.getLogger(__name__)
    if isinstance(array, da.core.Array):
        logger.info("Computing input dask array to Numpy memory-mapped array")
        array_path, _ = _dask_to_numpy_memmap(array, tmp_dir)
        logger.info(f"Numpy memory-mapped array is now at {array_path}")
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
    dask_method=None,
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
    parallel : bool or int or str, optional
        If `True`, the reprojection is carried out in parallel, and if a
        positive integer, this specifies the number of threads to use.
        The reprojection will be parallelized over output array blocks specified
        by ``block_size`` (if the block size is not set, it will be determined
        automatically). To use the currently active dask scheduler (e.g.
        dask.distributed), set this to ``'current-scheduler'``.
    reproject_func_kwargs : dict, optional
        Keyword arguments to pass through to ``reproject_func``
    return_type : {'numpy', 'dask' }, optional
        Whether to return numpy or dask arrays.
    dask_method : {'memmap', 'none'}, optional
        Method to use when input array is a dask array. The methods are:
            * ``'memmap'``: write out the entire input dask array to a temporary
              memory-mapped array. This requires enough disk space to store
              the entire input array, but should avoid accidentally loading
              the entire array into memory.
            * ``'none'``: load the dask array into memory as needed. This may
              result in the entire array being loaded into memory. However,
              this can be efficient under two conditions: if the array easily
              fits into memory (as this will then be faster than ``'memmap'``),
              and when the data contains more dimensions than the input WCS and
              the block_size is chosen to iterate over the extra dimensions.
    """

    logger = logging.getLogger(__name__)

    if return_type is None:
        return_type = "numpy"
    elif return_type not in ("numpy", "dask"):
        raise ValueError("return_type should be set to 'numpy' or 'dask'")

    if dask_method is None:
        dask_method = "memmap"
    elif dask_method not in ("memmap", "none"):
        raise ValueError("dask_method should be set to 'memmap' or 'none'")

    if reproject_func_kwargs is None:
        reproject_func_kwargs = {}

    # We set up a global temporary directory since this will be used e.g. to
    # store memory mapped Numpy arrays and zarr arrays.

    with tempfile.TemporaryDirectory() as local_tmp_dir:
        if array_out is None:
            if return_type != "dask":
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

            if return_type == "dask":
                raise ValueError(
                    "Output cannot be returned as dask arrays "
                    "when parallel=False and no block size has "
                    "been specified"
                )

            if isinstance(array_in, da.core.Array) and dask_method == "memmap":
                logger.info("Computing input dask array to Numpy memory-mapped array")
                array_path, array_in = _dask_to_numpy_memmap(array_in, local_tmp_dir)
                logger.info(f"Numpy memory-mapped array is now at {array_path}")

            logger.info(f"Calling {reproject_func.__name__} in non-dask mode")

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

        # Determine whether any broadcasting is taking place. This is when the
        # input/output WCS have fewer dimensions that the data, and any preceding
        # dimensions can be assumed to be independent of the WCS. At this point
        # shape_out will be the full size of the output array as this is updated
        # in parse_output_projection, even if shape_out was originally passed in as
        # the shape of a single image.
        broadcasting = wcs_in.low_level_wcs.pixel_n_dim < len(shape_out)

        logger.info(f"Broadcasting is {'' if broadcasting else 'not '}being used")

        # Check block size and determine whether block size indicates we should
        # parallelize over broadcasted dimension. The logic is as follows: if
        # the block size and output shape are the same size, then either the
        # block size should match the output shape along the broadcasted
        # dimensions or along the non-broadcasted dimensions. If it matches the
        # non-broadcasted dimensions we can parallelize over the broadcasted
        # dimensions. If the block size does not match the output shape, we
        # don't make any assumptions for now and assume a single chunk in the
        # missing dimensions.
        broadcasted_parallelization = False
        if broadcasting and block_size is not None and block_size != "auto":
            if len(block_size) == len(shape_out):
                if (
                    block_size[-wcs_in.low_level_wcs.pixel_n_dim :]
                    == shape_out[-wcs_in.low_level_wcs.pixel_n_dim :]
                ):
                    broadcasted_parallelization = True
                    block_size = (
                        block_size[: -wcs_in.low_level_wcs.pixel_n_dim]
                        + (-1,) * wcs_in.low_level_wcs.pixel_n_dim
                    )
                else:
                    for i in range(len(shape_out) - wcs_in.low_level_wcs.pixel_n_dim):
                        if block_size[i] != -1 and block_size[i] != shape_out[i]:
                            raise ValueError(
                                "block shape should either match output data shape along broadcasted dimension or non-broadcasted dimensions"
                            )
            elif len(block_size) < len(shape_out):
                block_size = [-1] * (len(shape_out) - len(block_size)) + list(block_size)
            else:
                raise ValueError(
                    f"block_size {len(block_size)} cannot have more elements "
                    f"than the dimensionality of the output ({len(shape_out)})"
                )

            # TODO: check for shape_out not matching shape_in along broadcasted dimensions

        logger.info(
            f"{'P' if broadcasted_parallelization else 'Not p'}arallelizing along "
            f"broadcasted dimension ({block_size=}, {shape_out=})"
        )

        if output_footprint is None and return_footprint:
            output_footprint = np.zeros(shape_out, dtype=float)

        shape_in = array_in.shape

        def reproject_single_block(a, array_or_path, block_info=None):

            if (
                a.ndim == 0
                or block_info is None
                or block_info == []
                or (isinstance(block_info, np.ndarray) and block_info.tolist() == [])
            ):
                return np.array([a, a])

            if isinstance(array_or_path, str) and array_or_path == "from-dict":
                array_or_path = dask_arrays["array"]

            # The WCS class from astropy is not thread-safe, see e.g.
            # https://github.com/astropy/astropy/issues/16244
            # https://github.com/astropy/astropy/issues/16245
            # To work around these issues, we make sure we do a deep copy of
            # the WCS object in here when using FITS WCS. This is a very fast
            # operation (<0.1ms) so should not be a concern in terms of
            # performance. We only need to do this for FITS WCS.

            wcs_in_cp = wcs_in.deepcopy() if isinstance(wcs_in, WCS) else wcs_in
            wcs_out_cp = wcs_out.deepcopy() if isinstance(wcs_out, WCS) else wcs_out

            slices = [
                slice(*x) for x in block_info[None]["array-location"][-wcs_out_cp.pixel_n_dim :]
            ]

            if isinstance(wcs_out, BaseHighLevelWCS):
                low_level_wcs = SlicedLowLevelWCS(wcs_out_cp.low_level_wcs, slices=slices)
            else:
                low_level_wcs = SlicedLowLevelWCS(wcs_out_cp, slices=slices)

            wcs_out_sub = HighLevelWCSWrapper(low_level_wcs)

            if isinstance(array_or_path, tuple):
                array_in = np.memmap(array_or_path[0], **array_or_path[1], mode="r")
            elif isinstance(array_or_path, str):
                array_in = np.memmap(array_or_path, dtype=float, shape=shape_in, mode="r")
            else:
                array_in = array_or_path

            if array_or_path is None:
                raise RuntimeError("array_or_path is not set")

            shape_out = block_info[None]["chunk-shape"][1:]

            array, footprint = reproject_func(
                array_in,
                wcs_in_cp,
                wcs_out_sub,
                shape_out=shape_out,
                array_out=np.zeros(shape_out),
                **reproject_func_kwargs,
            )

            return np.array([array, footprint])

        if broadcasted_parallelization:

            array_out_dask = da.empty(shape_out, chunks=block_size)
            if isinstance(array_in, da.core.Array):
                if array_in.chunksize != block_size:
                    logger.info(
                        f"Rechunking input dask array as chunks ({array_in.chunksize}) "
                        "do not match block size ({block_size})"
                    )
                    array_in = array_in.rechunk(block_size)
            else:

                class ArrayWrapper:

                    def __init__(self, array):
                        self._array = array
                        self.ndim = array.ndim
                        self.shape = array.shape
                        self.dtype = array.dtype

                    def __getitem__(self, item):
                        return self._array[item]

                array_in = da.asarray(
                    ArrayWrapper(array_in), name=str(uuid.uuid4()), chunks=block_size
                )

            result = da.map_blocks(
                reproject_single_block,
                array_out_dask,
                array_in,
                dtype="<f8",
                new_axis=0,
                chunks=((2,),) + array_out_dask.chunks,
            )

        else:

            # As we use the synchronous or threads scheduler, we don't need to worry about
            # the data getting copied, so if the data is already a Numpy array (including
            # a memory-mapped array) then we don't need to do anything special. However,
            # if the input array is a dask array, we should convert it to a Numpy
            # memory-mapped array so that it can be used by the various reprojection
            # functions (which don't internally work with dask arrays).

            if isinstance(array_in, np.memmap) and array_in.flags.c_contiguous:
                array_in_or_path = array_in.filename, {
                    "dtype": array_in.dtype,
                    "shape": array_in.shape,
                    "offset": array_in.offset,
                }
            elif isinstance(array_in, da.core.Array) or return_type == "dask":
                if return_type == "memmap":
                    if return_type == "dask":
                        # We should use a temporary directory that will persist beyond
                        # the call to the reproject function.
                        tmp_dir = tempfile.mkdtemp()
                    else:
                        tmp_dir = local_tmp_dir
                    array_in_or_path = as_delayed_memmap_path(_ArrayContainer(array_in), tmp_dir)
                else:
                    dask_arrays = {"array": array_in}
                    array_in_or_path = "from-dict"
            else:
                # Here we could set array_in_or_path to array_in_path if it has
                # been set previously, but in synchronous and threaded mode it is
                # better to simply pass a reference to the memmap array itself to
                # avoid having to load the memmap inside each
                # reproject_single_block call.
                array_in_or_path = array_in

            if block_size is not None and block_size != "auto":
                array_out_dask = da.empty(shape_out, chunks=block_size)
            else:
                if broadcasting:
                    chunks = (-1,) * (len(shape_out) - wcs_in.low_level_wcs.pixel_n_dim)
                    chunks += ("auto",) * wcs_in.low_level_wcs.pixel_n_dim
                    rechunk_kwargs = {"chunks": chunks}
                else:
                    rechunk_kwargs = {}
                array_out_dask = da.empty(shape_out)
                array_out_dask = array_out_dask.rechunk(
                    block_size_limit=64 * 1024**2, **rechunk_kwargs
                )

            logger.info("Setting up output dask array with map_blocks")

            result = da.map_blocks(
                reproject_single_block,
                array_out_dask,
                array_in_or_path,
                dtype="<f8",
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
            # will not work well in parallel mode when the destination is a
            # Numpy array. Instead, in this case we save the dask array to a zarr
            # array on disk which can be done in parallel, and re-load it as a dask
            # array. We can then use da.store in the next step using the
            # 'synchronous' scheduler since that is I/O limited so does not need
            # to be done in parallel.

            zarr_path = os.path.join(local_tmp_dir, f"{uuid.uuid4()}.zarr")

            logger.info(f"Computing output array directly to zarr array at {zarr_path}")

            if parallel == "current-scheduler":
                # Just use whatever is the current active scheduler, which can
                # be used for e.g. dask.distributed
                result.to_zarr(zarr_path)
            else:
                if isinstance(parallel, bool):
                    workers = {}
                else:
                    if parallel > 0:
                        workers = {"num_workers": parallel}
                    else:
                        raise ValueError(
                            "The number of processors to use must be strictly positive"
                        )

                with dask.config.set(scheduler="threads", **workers):
                    result.to_zarr(zarr_path)

            result = da.from_zarr(zarr_path)

        logger.info("Copying output zarr array into output Numpy arrays")

        if return_footprint:
            da.store(
                [result[0], result[1]],
                [array_out, output_footprint],
                compute=True,
                scheduler="synchronous",
            )
            output = array_out, output_footprint
        else:
            da.store(
                result[0],
                array_out,
                compute=True,
                scheduler="synchronous",
            )
            output = array_out

    if return_footprint:
        return output[0], output[1]
    else:
        return output

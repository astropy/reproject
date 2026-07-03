import logging
import mmap
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

from ._array_utils import ArrayWrapper
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
    non_reprojected_dims=None,
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
    non_reprojected_dims : tuple, optional
        Leading dimensions of the data that should not be reprojected but for
        which a one-to-one mapping between input and output pixels is assumed,
        given as a tuple of sequential integers starting from zero (e.g.
        ``(0,)`` or ``(0, 1)``). If `None` (the default), any leading dimensions
        for which the WCS has fewer dimensions than the data are treated this
        way. Reprojecting fewer dimensions than the WCS currently requires an
        explicit ``block_size``; its entries along the reprojected dimensions
        may either match the output shape or be smaller, in which case each
        non-reprojected slice is reprojected in sub-tiles of that size.
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

    # For now, we are quite restrictive in what non_reprojected_dims can
    # be, but it is designed so that if we wanted we could support more use
    # cases in future. For now, it has to be a tuple where each element is
    # sequential from zero, e.g. (0,) or (0, 1) or (0, 1, 2)

    if non_reprojected_dims is None:
        n_dim_reproject = min(wcs_in.low_level_wcs.pixel_n_dim, wcs_out.low_level_wcs.pixel_n_dim)
    else:
        if non_reprojected_dims != tuple(range(len(non_reprojected_dims))):
            raise ValueError(
                "non_reprojected_dims should be a tuple with values increasing sequentially from zero"
            )
        # If either WCS already has fewer dimensions than the data, the missing
        # dimensions are implicitly non-reprojected, so the shortfall has to be
        # consistent with the number of non_reprojected_dims requested.
        for label, wcs in (("input", wcs_in), ("output", wcs_out)):
            n_dim_missing = len(shape_out) - wcs.low_level_wcs.pixel_n_dim
            if n_dim_missing > 0 and n_dim_missing != len(non_reprojected_dims):
                raise ValueError(
                    f"The {label} WCS has {wcs.low_level_wcs.pixel_n_dim} pixel dimensions "
                    f"which is fewer than the {len(shape_out)} data dimensions, but the "
                    f"difference ({n_dim_missing}) does not match the number of "
                    f"non_reprojected_dims ({len(non_reprojected_dims)})"
                )
        n_dim_reproject = len(shape_out) - len(non_reprojected_dims)
        if n_dim_reproject < 1:
            raise ValueError(
                "non_reprojected_dims should leave at least one dimension to be " "reprojected"
            )

    # ``wcs_slicing_required`` flags that we are reprojecting fewer dimensions than
    # the input or output WCS describes, so the WCS must be sliced down to the
    # reprojected dimensions for each non-reprojected slice. That slicing is only
    # implemented on the path that parallelizes over the non-reprojected
    # (broadcasted) dimensions; the other code paths raise NotImplementedError below
    # rather than attempting it. It is gated on non_reprojected_dims being set, the
    # only way to opt into reprojecting fewer dimensions than the WCS; a plain
    # mismatch between the input and output WCS dimensionality is instead a
    # validation error raised by the underlying reprojection function.
    wcs_slicing_required = non_reprojected_dims is not None and (
        n_dim_reproject < wcs_in.low_level_wcs.pixel_n_dim
        or n_dim_reproject < wcs_out.low_level_wcs.pixel_n_dim
    )

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

        if block_size is None and parallel is False and not wcs_slicing_required:
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

            logger.debug(f"Calling {reproject_func.__name__} in non-dask mode")

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
        broadcasting = n_dim_reproject < len(shape_out)

        logger.info(
            f"Broadcasting is {'' if broadcasting else 'not '}being used, "
            f"reprojecting last {n_dim_reproject} axes"
        )

        # The output shape must match the input shape along any non-reprojected
        # (broadcasted) dimensions.

        shape_in = array_in.shape
        shape_out = tuple(shape_out)

        if shape_out[:-n_dim_reproject] != shape_in[:-n_dim_reproject]:
            raise ValueError("Input shape should match output shape for non-reprojected dimensions")

        # If an explicit block size was passed, normalize it to have the same
        # number of elements as shape_out, expanding it if it only covers the
        # reprojected dimensions and replacing any -1 values by the full size
        # along the corresponding dimension. If block_size is None or 'auto',
        # the chunking is determined automatically further below.

        if block_size is not None and block_size != "auto":

            if len(block_size) > len(shape_out):
                raise ValueError(
                    f"block_size {block_size} cannot have more elements "
                    f"than the dimensionality of the output ({len(shape_out)})"
                )

            if len(block_size) != n_dim_reproject and len(block_size) != len(shape_out):
                raise ValueError(
                    f"block_size {block_size} should have either "
                    f"{n_dim_reproject} or {len(shape_out)} elements"
                )

            if len(block_size) == n_dim_reproject:
                block_size = (-1,) * (len(shape_out) - n_dim_reproject) + tuple(block_size)

            block_size = tuple(
                block_size[i] if block_size[i] != -1 else shape_out[i]
                for i in range(len(block_size))
            )

        # Decide whether the requested block size means we should parallelize over
        # the broadcasted (non-reprojected, leading) dimensions. block_size has
        # already been padded above to one entry per output dimension, so this is not
        # about the number of entries but about which dimensions the block spans the
        # full output extent along:
        #  - if the block spans the full extent along the reprojected (trailing)
        #    dimensions, each block is one whole non-reprojected slice, so we
        #    parallelize over the broadcasted dimensions (one slice per block);
        #  - if instead it spans the full extent along the broadcasted (leading)
        #    dimensions, the block tiles the reprojected dimensions and we do not
        #    parallelize over the broadcasted dimensions;
        #  - if it spans the full extent along neither, we raise, unless
        #    non_reprojected_dims requires slicing the WCS per non-reprojected slice,
        #    in which case a block smaller than the slice sub-tiles each slice.
        broadcasted_parallelization = False
        if broadcasting and block_size is not None and block_size != "auto":
            if block_size[-n_dim_reproject:] == shape_out[-n_dim_reproject:]:
                broadcasted_parallelization = True
            elif wcs_slicing_required:
                # A block smaller than the output along the reprojected dimensions
                # is only meaningful when the WCS has to be sliced per broadcasted
                # slice (i.e. non_reprojected_dims). We parallelize one broadcasted
                # slice per block and let dask additionally tile the reprojected
                # dimensions according to the block size, which bounds the
                # coordinate-transform memory (it would otherwise scale with the
                # full slice size). Each output tile is still reprojected from the
                # whole input slice, since any output pixel can map anywhere within
                # it.
                broadcasted_parallelization = True
            elif block_size[:-n_dim_reproject] != shape_out[:-n_dim_reproject]:
                raise ValueError(
                    "block shape should either match output data shape along "
                    "reprojected dimensions or non-reprojected dimensions"
                )
            if broadcasted_parallelization:
                # One broadcasted slice per block; dask tiles the reprojected
                # dimensions using whatever block size was requested along them.
                # The block size along the non-reprojected dimensions must be 1
                # or span the full extent (equivalent here, since blocks are
                # single slices either way); anything else would be silently
                # reinterpreted, so raise instead.
                if any(
                    entry not in (1, shape_out[idim])
                    for idim, entry in enumerate(block_size[: len(shape_out) - n_dim_reproject])
                ):
                    raise ValueError(
                        f"block_size {block_size} should be 1 or match the output shape "
                        "along the non-reprojected dimensions (each block covers a "
                        "single non-reprojected slice)"
                    )
                block_size = (1,) * (len(shape_out) - n_dim_reproject) + block_size[
                    -n_dim_reproject:
                ]

        logger.info(
            f"{'P' if broadcasted_parallelization else 'Not p'}arallelizing along "
            f"broadcasted dimension ({block_size=}, {shape_out=})"
        )

        # TODO: support block_size="auto" (and the default of None) together
        # with non_reprojected_dims so that this does not have to raise; "auto"
        # currently falls through to the generic auto-chunking path further
        # below, which cannot parallelize over the non-reprojected dimensions.
        if wcs_slicing_required and not broadcasted_parallelization:
            raise NotImplementedError(
                "Reprojecting fewer dimensions than the input or output WCS "
                "(for example using non_reprojected_dims) currently requires "
                "passing an explicit block_size whose entries along the reprojected "
                "dimensions either match the output shape or are smaller (in which "
                "case each non-reprojected slice is reprojected in sub-tiles of "
                "that size), "
                "optionally with parallel=True to compute the blocks concurrently"
            )

        if output_footprint is None and return_footprint and return_type != "dask":
            output_footprint = np.zeros(shape_out, dtype=float)

        def reproject_single_block(a, array_or_path, block_info=None):

            if (
                a.ndim == 0
                or block_info is None
                or block_info == []
                or (isinstance(block_info, np.ndarray) and block_info.tolist() == [])
            ):
                return np.array([a, a])

            if isinstance(array_or_path, _ArrayContainer):
                array_or_path = array_or_path._array

            shape_out = block_info[None]["chunk-shape"][1:]

            # The WCS class from astropy is not thread-safe, see e.g.
            # https://github.com/astropy/astropy/issues/16244
            # https://github.com/astropy/astropy/issues/16245
            # To work around these issues, we make sure we do a deep copy of
            # the WCS object in here when using FITS WCS. This is a very fast
            # operation (<0.1ms) so should not be a concern in terms of
            # performance. We only need to do this for FITS WCS.

            wcs_in_cp = wcs_in.deepcopy() if isinstance(wcs_in, WCS) else wcs_in
            wcs_out_cp = wcs_out.deepcopy() if isinstance(wcs_out, WCS) else wcs_out

            # Along the reprojected dimensions the input is always kept whole (any
            # output pixel can map anywhere within it) while dask may tile the
            # output; along the broadcasted dimensions each block is a single
            # slice. slices_in/slices_out reduce the input/output WCS to this
            # block; the matching broadcasted slice of the input either arrives as
            # the aligned input block or, when the input was passed whole (lazy
            # dask input), is read out below using slices_in_data.
            slices_in = []
            slices_out = []
            slices_in_data = []
            for idx in range(len(shape_out)):
                interval = block_info[None]["array-location"][idx + 1]
                if broadcasted_parallelization and idx < len(shape_out) - n_dim_reproject:
                    if interval[1] - interval[0] != 1:
                        raise RuntimeError(
                            f"Expected a chunk of width 1 along dimension {idx} "
                            f"(got {interval[1] - interval[0]})"
                        )
                    slices_in.append(interval[0])
                    slices_out.append(interval[0])
                    slices_in_data.append(slice(*interval))
                else:
                    slices_in.append(slice(None))
                    slices_out.append(slice(*block_info[None]["array-location"][idx + 1]))
                    slices_in_data.append(slice(None))

            slices_in = slices_in[-wcs_in.low_level_wcs.pixel_n_dim :]
            slices_out = slices_out[-wcs_out.low_level_wcs.pixel_n_dim :]

            if broadcasted_parallelization:
                if isinstance(wcs_in_cp, BaseHighLevelWCS):
                    low_level_wcs_in = SlicedLowLevelWCS(wcs_in_cp.low_level_wcs, slices=slices_in)
                else:
                    low_level_wcs_in = SlicedLowLevelWCS(wcs_in_cp, slices=slices_in)

                wcs_in_sub = HighLevelWCSWrapper(low_level_wcs_in)
            else:
                wcs_in_sub = wcs_in_cp

            if isinstance(wcs_out_cp, BaseHighLevelWCS):
                low_level_wcs_out = SlicedLowLevelWCS(wcs_out_cp.low_level_wcs, slices=slices_out)
            else:
                low_level_wcs_out = SlicedLowLevelWCS(wcs_out_cp, slices=slices_out)

            wcs_out_sub = HighLevelWCSWrapper(low_level_wcs_out)

            if broadcasted_parallelization and input_aligned:
                # The input was passed as an aligned dask array, so array_or_path
                # is already this block's broadcasted slice of the input, kept
                # whole along the reprojected dimensions (see above).
                array_in = array_or_path
            elif isinstance(array_or_path, tuple):
                array_in = np.memmap(array_or_path[0], **array_or_path[1], mode="r")
            elif isinstance(array_or_path, str):
                array_in = np.memmap(array_or_path, dtype=float, shape=shape_in, mode="r")
            else:
                array_in = array_or_path

            if array_in is None:
                raise RuntimeError("array_or_path is not set")

            if broadcasted_parallelization and not input_aligned:
                # The input was passed whole as a lazy dask array; read out a lazy
                # view of this block's broadcasted slice so a streaming
                # reprojection core only computes the input chunks it touches.
                array_in = array_in[tuple(slices_in_data)]

            array, footprint = reproject_func(
                array_in,
                wcs_in_sub,
                wcs_out_sub,
                shape_out=shape_out,
                array_out=np.zeros(shape_out),
                **reproject_func_kwargs,
            )

            return np.array([array, footprint])

        input_aligned = False
        if broadcasted_parallelization and (
            not isinstance(array_in, da.core.Array)
            or dask_method != "none"
            or all(len(chunks) == 1 for chunks in array_in.chunks[-n_dim_reproject:])
        ):
            # Pass the input as a second dask array with one chunk per broadcasted
            # slice, kept whole along the reprojected dimensions (any output pixel
            # can map anywhere within its slice). map_blocks broadcasts the single
            # chunk along the reprojected dimensions to every output tile of that
            # slice, so each slice is computed exactly once and streamed to the
            # tasks that need it: dask array inputs are never materialized in
            # full, sub-tiled slices do not recompute their input per tile, and
            # under a distributed scheduler each task depends only on its own
            # slice rather than embedding the whole input. The exception is a dask
            # input with dask_method='none' that is chunked below one slice along
            # the reprojected dimensions: materializing it here would forgo the
            # ability of streaming reprojection cores to work chunk by chunk
            # without ever holding a whole slice, so it is kept lazy below.
            input_aligned = True
            input_chunks = (1,) * (array_in.ndim - n_dim_reproject) + (-1,) * n_dim_reproject
            if isinstance(array_in, da.core.Array):
                array_in_dask = array_in.rechunk(input_chunks)
                # Blockwise fusion would fold the input graph into every output
                # tile task, recomputing each broadcasted slice once per tile of
                # that slice; routing each slice through a delayed task pins it
                # as a single node in the graph that all of its tiles share.
                delayed_blocks = array_in_dask.to_delayed()
                pieces = np.empty(delayed_blocks.shape, dtype=object)
                for index in np.ndindex(delayed_blocks.shape):
                    shape = tuple(
                        array_in_dask.chunks[idim][index[idim]]
                        for idim in range(array_in_dask.ndim)
                    )
                    pieces[index] = da.from_delayed(
                        delayed_blocks[index], shape=shape, dtype=array_in_dask.dtype
                    )
                array_in_or_path = da.block(pieces.tolist())
            else:
                # ArrayWrapper (plus the explicit name) prevents dask from
                # hashing the whole buffer to name the array, which for a memmap
                # would silently load the entire file into memory (see
                # https://github.com/dask/dask/issues/11850).
                array_in_or_path = da.from_array(
                    ArrayWrapper(array_in),
                    name=f"reproject-input-{uuid.uuid4().hex}",
                    chunks=input_chunks,
                )

        elif broadcasted_parallelization:
            # A dask input with dask_method='none' chunked below one slice along
            # the reprojected dimensions: pass it whole as an opaque constant and
            # let each block read out a lazy view of its own slice, so that a
            # streaming reprojection core (e.g. interpolation via dask-image) only
            # ever computes the input chunks that each output tile touches and a
            # full slice need never be materialized at once. The tradeoff is that
            # input chunks touched by several tiles are computed once per tile.
            array_in_or_path = _ArrayContainer(array_in)

        # For the remaining (non-broadcasted) cases the input is passed to
        # map_blocks as an opaque (non-dask) argument, so that every task sees the
        # whole input. As we use the synchronous or threads scheduler, we don't need
        # to worry about the data getting copied, so if the data is already a Numpy
        # array (including a memory-mapped array) then we don't need to do anything
        # special. However, if the input array is a dask array, we should convert
        # it to a Numpy memory-mapped array so that it can be used by the various
        # reprojection functions (which don't internally work with dask arrays).

        # Only base memmaps can be reconstructed from filename and offset: views
        # (e.g. a slice of a memmap) keep the parent's unadjusted .offset, so
        # reconstructing them would silently read the wrong file region. Views
        # fall through and are passed by reference like plain arrays.
        elif (
            isinstance(array_in, np.memmap)
            and array_in.flags.c_contiguous
            and isinstance(array_in.base, mmap.mmap)
        ):
            array_in_or_path = array_in.filename, {
                "dtype": array_in.dtype,
                "shape": array_in.shape,
                "offset": array_in.offset,
            }
        elif isinstance(array_in, da.core.Array) or return_type == "dask":
            if dask_method == "memmap":
                if return_type == "dask":
                    # We should use a temporary directory that will persist beyond
                    # the call to the reproject function.
                    tmp_dir = tempfile.mkdtemp()
                else:
                    tmp_dir = local_tmp_dir
                array_in_or_path = as_delayed_memmap_path(_ArrayContainer(array_in), tmp_dir)
            else:
                # Wrap the dask array in _ArrayContainer so dask treats it as an
                # opaque constant (rather than a collection to compute/align) when
                # it is passed through to the block function.
                array_in_or_path = _ArrayContainer(array_in)
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
                chunks = (-1,) * (len(shape_out) - n_dim_reproject)
                chunks += ("auto",) * n_dim_reproject
                rechunk_kwargs = {"chunks": chunks}
            else:
                rechunk_kwargs = {}
            array_out_dask = da.empty(shape_out)
            array_out_dask = array_out_dask.rechunk(block_size_limit=64 * 1024**2, **rechunk_kwargs)

        logger.info("Setting up output dask array with map_blocks")

        # Declare the exact (possibly ragged) chunks of the output template so
        # that edge blocks are computed at their true size rather than being
        # reprojected at the full block size and truncated afterwards.
        result = da.map_blocks(
            reproject_single_block,
            array_out_dask,
            array_in_or_path,
            dtype="<f8",
            new_axis=0,
            chunks=((2,),) + array_out_dask.chunks,
        )

        # Ensure that there are no more references to Numpy memmaps
        array_in = None
        array_in_or_path = None

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

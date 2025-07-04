import numpy as np

__all__ = ["map_coordinates", "sample_array_edges"]


def find_chunk_shape(shape, max_chunk_size=None):
    """
    Given the shape of an n-dimensional array, and the maximum number of
    elements in a chunk, return the largest chunk shape to use for iteration.

    This currently assumes the optimal chunk shape to return is for C-contiguous
    arrays.

    Parameters
    ----------
    shape : iterable
        The shape of the n-dimensional array.
    max_chunk_size : int, optional
        The maximum number of elements per chunk.
    """

    if max_chunk_size is None:
        return tuple(shape)

    block_shape = []

    max_repeat_remaining = max_chunk_size

    for size in shape[::-1]:
        if max_repeat_remaining > size:
            block_shape.append(size)
            max_repeat_remaining = max_repeat_remaining // size
        else:
            block_shape.append(max_repeat_remaining)
            max_repeat_remaining = 1

    return tuple(block_shape[::-1])


def iterate_chunks(shape, *, max_chunk_size):
    """
    Given a data shape and a chunk shape (or maximum chunk size), iteratively
    return slice objects that can be used to slice the array.

    Parameters
    ----------
    shape : iterable
        The shape of the n-dimensional array.
    max_chunk_size : int
        The maximum number of elements per chunk.
    """

    if np.prod(shape) == 0:
        return

    chunk_shape = find_chunk_shape(shape, max_chunk_size)

    ndim = len(chunk_shape)
    start_index = [0] * ndim

    shape = list(shape)

    while start_index <= shape:
        end_index = [min(start_index[i] + chunk_shape[i], shape[i]) for i in range(ndim)]

        slices = tuple([slice(start_index[i], end_index[i]) for i in range(ndim)])

        yield slices

        # Update chunk index. What we do is to increment the
        # counter for the first dimension, and then if it
        # exceeds the number of elements in that direction,
        # cycle back to zero and advance in the next dimension,
        # and so on.
        start_index[0] += chunk_shape[0]
        for i in range(ndim - 1):
            if start_index[i] >= shape[i]:
                start_index[i] = 0
                start_index[i + 1] += chunk_shape[i + 1]

        # We can now check whether the iteration is finished
        if start_index[-1] >= shape[-1]:
            break


def at_least_float32(array):
    if array.dtype.kind == "f" and array.dtype.itemsize >= 4:
        return array
    else:
        return array.astype(np.float32)


def memory_efficient_access(array, chunk):
    # If we access a number of chunks from a memory-mapped array, memory usage
    # will increase and could crash e.g. dask.distributed workers. We therefore
    # use a temporary memmap to load the data.
    if isinstance(array, np.memmap) and array.flags.c_contiguous:
        array_tmp = np.memmap(
            array.filename,
            mode="r",
            dtype=array.dtype,
            shape=array.shape,
            offset=array.offset,
        )
        return array_tmp[chunk]
    else:
        return array[chunk]


def map_coordinates(
    image, coords, max_chunk_size=None, output=None, optimize_memory=False, **kwargs
):
    # In the built-in scipy map_coordinates, the values are defined at the
    # center of the pixels. This means that map_coordinates does not
    # correctly treat pixels that are in the outer half of the outer pixels.
    # We solve this by resetting any coordinates that are in the outer half of
    # the border pixels to be at the center of the border pixels. We used to
    # instead pad the array but this was not memory efficient as it ended up
    # producing a copy of the output array.

    # In addition, map_coordinates is not efficient when given big-endian Numpy
    # arrays as it will then make a copy, which is an issue when dealing with
    # memory-mapped FITS files that might be larger than memory. Therefore, for
    # big-endian arrays, we operate in chunks with a size smaller or equal to
    # max_chunk_size.

    # The optimize_memory option isn't used right not by the rest of reproject
    # but it is a mode where if we are in a memory-constrained environment, we
    # re-create memmaps for individual chunks to avoid caching the whole array.
    # We need to decide how to expose this to users.

    # TODO: check how this should behave on a big-endian system.

    from scipy.ndimage import map_coordinates as scipy_map_coordinates

    original_shape = image.shape

    # We copy the coordinates array as we then modify it in-place below to clip
    # to the edges of the array.

    coords = coords.copy()
    for i in range(coords.shape[0]):
        coords[i][(coords[i] < 0) & (coords[i] >= -0.5)] = 0
        coords[i][(coords[i] < original_shape[i] - 0.5) & (coords[i] >= original_shape[i] - 1)] = (
            original_shape[i] - 1
        )

    # If the data type is native and we are not doing spline interpolation,
    # then scipy_map_coordinates deals properly with memory maps, so we can use
    # it without chunking. Otherwise, we need to iterate over data chunks.
    if image.dtype.isnative and "order" in kwargs and kwargs["order"] <= 1:
        values = scipy_map_coordinates(at_least_float32(image), coords, output=output, **kwargs)
    else:
        if output is None:
            output = np.repeat(np.nan, coords.shape[1])

        values = output

        include = np.ones(coords.shape[1], dtype=bool)

        if "order" in kwargs and kwargs["order"] <= 1:
            padding = 1
        else:
            padding = 10

        for chunk in iterate_chunks(image.shape, max_chunk_size=max_chunk_size):

            include[...] = True
            for idim, slc in enumerate(chunk):
                include[(coords[idim] < slc.start) | (coords[idim] >= slc.stop)] = False

            if not np.any(include):
                continue

            chunk = list(chunk)

            # Adjust chunks to add padding
            for idim, slc in enumerate(chunk):
                start = max(0, slc.start - padding)
                stop = min(original_shape[idim], slc.stop + padding)
                chunk[idim] = slice(start, stop)

            chunk = tuple(chunk)

            coords_subset = coords[:, include].copy()
            for idim, slc in enumerate(chunk):
                coords_subset[idim, :] -= slc.start

            if optimize_memory:
                image_subset = memory_efficient_access(image, chunk)
            else:
                image_subset = image[chunk]

            output[include] = scipy_map_coordinates(
                at_least_float32(image_subset), coords_subset, **kwargs
            )

    reset = np.zeros(coords.shape[1], dtype=bool)

    for i in range(coords.shape[0]):
        reset |= coords[i] < -0.5
        reset |= coords[i] > original_shape[i] - 0.5

    values[reset] = kwargs.get("cval", 0.0)

    return values


def sample_array_edges(shape, *, n_samples):
    # Given an N-dimensional array shape, sample each edge of the array using
    # the requested number of samples (which will include vertices). To do this
    # we iterate through the dimensions and for each one we sample the points
    # in that dimension and iterate over the combination of other vertices.
    # Returns an array with dimensions (N, n_samples)
    all_positions = []
    ndim = len(shape)
    shape = np.array(shape)
    for idim in range(ndim):
        for vertex in range(2**ndim):
            positions = -0.5 + shape * ((vertex & (2 ** np.arange(ndim))) > 0).astype(int)
            positions = np.broadcast_to(positions, (n_samples, ndim)).copy()
            positions[:, idim] = np.linspace(-0.5, shape[idim] - 0.5, n_samples)
            all_positions.append(positions)
    positions = np.unique(np.vstack(all_positions), axis=0).T
    return positions

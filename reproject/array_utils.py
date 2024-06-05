import numpy as np

__all__ = ["map_coordinates", "sample_array_edges"]


def map_coordinates(image, coords, **kwargs):
    # In the built-in scipy map_coordinates, the values are defined at the
    # center of the pixels. This means that map_coordinates does not
    # correctly treat pixels that are in the outer half of the outer pixels.
    # We solve this by resetting any coordinates that are in the outer half of
    # the border pixels to be at the center of the border pixels. We used to
    # instead pad the array but this was not memory efficient as it ended up
    # producing a copy of the output array.

    from scipy.ndimage import map_coordinates as scipy_map_coordinates

    original_shape = image.shape

    # We copy the coordinates array as we then modify it in-place below
    coords = coords.copy()
    for i in range(coords.shape[0]):
        coords[i][(coords[i] < 0) & (coords[i] >= -0.5)] = 0
        coords[i][(coords[i] < original_shape[i] - 0.5) & (coords[i] >= original_shape[i] - 1)] = (
            original_shape[i] - 1
        )

    values = scipy_map_coordinates(image, coords, **kwargs)

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

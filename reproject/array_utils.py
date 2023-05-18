import numpy as np

__all__ = ["map_coordinates"]


def map_coordinates(image, coords, **kwargs):
    # In the built-in scipy map_coordinates, the values are defined at the
    # center of the pixels. This means that map_coordinates does not
    # correctly treat pixels that are in the outer half of the outer pixels.
    # We solve this by extending the array, updating the pixel coordinates,
    # then getting rid of values that were sampled in the range -1 to -0.5
    # and n to n - 0.5.

    from scipy.ndimage import map_coordinates as scipy_map_coordinates

    original_shape = image.shape

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

import numpy as np

__all__ = ['match_backgrounds_inplace']


def match_backgrounds_inplace(arrays):

    N = len(arrays)

    # Set up matrix to record differences
    offsets = np.ones((N, N)) * np.nan

    # Loop over all pairs of images and check for overlap
    for i1, array1 in enumerate(arrays):
        for i2, array2 in enumerate(arrays):
            if i2 <= i1:
                continue
            if array1.overlaps(array2):
                difference = array1 - array2
                if np.any(difference.footprint):
                    values = difference.array[difference.footprint]
                    offsets[i1, i2] = np.median(values)
                    offsets[i2, i1] = -offsets[i1, i2]

    # We now need to iterate to find an optimal solution to the offsets

    inds = np.arange(N)
    b = np.zeros(N)
    eta0 = 1. / N  # Initial learning rate.

    red = 1.

    for main_iter in range(10000):

        if main_iter > 0 and main_iter % 500 == 0:
            red /= 2.

        np.random.shuffle(inds)

        # Update learning rate
        eta = eta0 * red

        for i in inds:

            if np.isnan(b[i]):
                continue

            keep = ~np.isnan(offsets[i, :])
            b[i] += eta * np.sum(offsets[i, keep]
                                 - b[i, np.newaxis]
                                 + b[keep][np.newaxis, :])

        mn = np.mean(b[~np.isnan(b)])
        b -= mn

    for array, offset in zip(arrays, b):
        array.array -= offset

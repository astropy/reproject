# Licensed under a 3-clause BSD style license - see LICENSE.rst

from math import exp

import numpy as np

__all__ = ["solve_corrections_sgd", "determine_offset_matrix"]


def determine_offset_matrix(arrays):
    """
    Given a list of ReprojectedArraySubset, determine the offset
    matrix between all arrays.

    Parameters
    ----------
    arrays : list
        The list of ReprojectedArraySubset objects to determine the offsets for.

    Returns
    -------
    `numpy.ndarray`
        The offset matrix.
    """

    N = len(arrays)

    # Set up matrix to record differences
    offset_matrix = np.ones((N, N)) * np.nan

    # Loop over all pairs of images and check for overlap
    for i1, array1 in enumerate(arrays):
        for i2, array2 in enumerate(arrays):
            if i2 <= i1:
                continue
            if array1.overlaps(array2):
                difference = array1 - array2
                if np.any(difference.footprint):
                    values = difference.array[difference.footprint]
                    offset_matrix[i1, i2] = np.median(values)
                    offset_matrix[i2, i1] = -offset_matrix[i1, i2]

    return offset_matrix


def solve_corrections_sgd(offset_matrix, eta_initial=1, eta_half_life=100, rtol=1e-10, atol=0):
    r"""
    Given a matrix of offsets from each image to each other image, find the
    optimal offsets to use using Stochastic Gradient Descent.

    Given N images, we can construct an NxN matrix Oij giving the typical (e.g.
    mean, median, or other statistic) offset from each image to each other
    image. This can be a reasonably sparse matrix since not all images
    necessarily overlap. From this we then want to find a vector of N
    corrections Ci to apply to each image to minimize the differences.

    We do this by using the Stochastic Gradient Descent algorithm:

    https://en.wikipedia.org/wiki/Stochastic_gradient_descent

    Essentially what we are trying to minimize is the difference between Dij
    and a matrix of the same shape constructed from the Oi values.

    The learning rate is decreased using a decaying exponential:

        $$\eta = \eta_{\rm initial} * \exp{(-i/t_{\eta})}$$

    Parameters
    ----------
    offset_matrix : `~numpy.ndarray`
        The NxN matrix giving the offsets between all images (or NaN if
        an offset could not be determined).
    eta_initial : float
        The initial learning rate to use.
    eta_half_life : float
        The number of iterations after which the learning rate should be
        decreased by a factor $e$.
    rtol : float
        The relative tolerance to use to determine if the corrections have
        converged.
    atol : float
        The absolute tolerance to use to determine if the corrections have
        converged.

    Returns
    -------
    `numpy.ndarray`
        The corrections for each frame.
    """

    if offset_matrix.ndim != 2 or offset_matrix.shape[0] != offset_matrix.shape[1]:
        raise ValueError("offset_matrix should be a square NxN matrix")

    N = offset_matrix.shape[0]

    indices = np.arange(N)
    corrections = np.zeros(N)

    # Keep track of previous corrections to know whether the algorithm
    # has converged
    previous_corrections = None

    for iteration in range(int(eta_half_life * 10)):
        # Shuffle the indices to avoid cyclical behavior
        np.random.shuffle(indices)

        # Update learning rate
        eta = eta_initial * exp(-iteration / eta_half_life)

        # Loop over each index and update the offset. What we call rows and
        # columns is arbitrary, but for the purpose of the comments below, we
        # treat this as iterating over rows of the matrix.
        for i in indices:
            if np.isnan(corrections[i]):
                continue

            # Since the matrix might be sparse, we consider only columns which
            # are not NaN
            keep = ~np.isnan(offset_matrix[i, :])

            # Compute the row of the offset matrix one would get with the
            # current offsets
            fitted_offset_matrix_row = corrections[i] - corrections[keep]

            # The difference between the actual row in the matrix and this
            # fitted row gives us a measure of the gradient, so we then
            # adjust the solution in that direction.
            corrections[i] += eta * np.mean(offset_matrix[i, keep] - fitted_offset_matrix_row)

        # Subtract the mean offset from the offsets to make sure that the
        # corrections stay centered around zero
        corrections -= np.nanmean(corrections)

        if previous_corrections is not None:
            if np.allclose(corrections, previous_corrections, rtol=rtol, atol=atol):
                break  # the algorithm has converged

        previous_corrections = corrections.copy()

    return corrections

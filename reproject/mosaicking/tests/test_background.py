# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest
from numpy.testing import assert_allclose

from ..background import solve_corrections_sgd

# Try and cover a range of matrix sizes and absolute scales of corrections
CASES = [(4, 1.),
         (33, 1e30),
         (44, 1e-50),
         (132, 1e10),
         (1441, 1e-5)]


@pytest.mark.parametrize(('N', 'scale'), CASES)
def test_solve_corrections_sgd(N, scale):

    # Generate random corrections
    expected = np.random.uniform(-scale, scale, N)

    # Generate offsets matrix
    offset_matrix = expected[:, np.newaxis] - expected[np.newaxis, :]

    # Add some NaN values
    offset_matrix[1, 2] = np.nan

    # Determine corrections
    actual = solve_corrections_sgd(offset_matrix)

    # Compare the mean-subtracted corrections since there might be an
    # arbitrary offset
    assert_allclose(actual - np.mean(actual), expected - np.mean(expected))

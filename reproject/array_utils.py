# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Backward-compatibility shim. The array_utils module is now private.
Import from reproject directly or use reproject._array_utils.
"""
import warnings

from astropy.utils.exceptions import AstropyDeprecationWarning

warnings.warn(
    "reproject.array_utils is a private module and should not be imported "
    "directly. Please use the public API from reproject instead.",
    AstropyDeprecationWarning,
    stacklevel=2,
)

from ._array_utils import *  # noqa: E402, F401, F403
from ._array_utils import (  # noqa: E402, F401
    ArrayWrapper,
    at_least_float32,
    dask_map_coordinates,
    find_chunk_shape,
    iterate_chunks,
    map_coordinates,
    memory_efficient_access,
    sample_array_edges,
)

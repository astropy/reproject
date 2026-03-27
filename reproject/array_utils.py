# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Backward-compatibility shim. The array_utils module is now private.
Import from reproject directly or use reproject._array_utils.
"""

import warnings

from astropy.utils.exceptions import AstropyDeprecationWarning

warnings.warn(
    "reproject.array_utils is a private module and will be removed in a future "
    "version.  Please use the public API from reproject instead.",
    AstropyDeprecationWarning,
    stacklevel=2,
)

from ._array_utils import *  # noqa: E402, F401, F403

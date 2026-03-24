# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Backward-compatibility shim. The wcs_utils module is now private.
Import from reproject directly or use reproject._wcs_utils.
"""
import warnings

from astropy.utils.exceptions import AstropyDeprecationWarning

warnings.warn(
    "reproject.wcs_utils is a private module and should not be imported "
    "directly. Please use the public API from reproject instead.",
    AstropyDeprecationWarning,
    stacklevel=2,
)

from ._wcs_utils import *  # noqa: E402, F401, F403

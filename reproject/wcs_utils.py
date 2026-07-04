# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Backward-compatibility shim. The wcs_utils module is now private.
Import from reproject directly or use reproject._wcs_utils.
"""

import warnings

from astropy.utils.exceptions import AstropyDeprecationWarning

from ._wcs_utils import has_celestial, pixel_scale
from ._wcs_utils import pixel_to_pixel_chunked as _pixel_to_pixel_chunked

warnings.warn(
    "reproject.wcs_utils is a private module and will be removed in a future "
    "version.  Please use the public API from reproject instead.",
    AstropyDeprecationWarning,
    stacklevel=2,
)

# pixel_to_pixel_with_roundtrip has been replaced by pixel_to_pixel_chunked but
# is kept here for backward-compatibility until this deprecated module is
# removed. The new function is intentionally not re-exported through this shim.
__all__ = ["has_celestial", "pixel_scale", "pixel_to_pixel_with_roundtrip"]


def pixel_to_pixel_with_roundtrip(wcs1, wcs2, *inputs):
    return _pixel_to_pixel_chunked(wcs1, wcs2, *inputs, roundtrip=True)

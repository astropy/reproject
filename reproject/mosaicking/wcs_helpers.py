# Licensed under a 3-clause BSD style license - see LICENSE.rst
import warnings

from astropy.utils.exceptions import AstropyDeprecationWarning

warnings.warn(
    "reproject.mosaicking.wcs_helpers is a private module and should not be imported "
    "directly. Please use the public API from reproject.mosaicking instead.",
    AstropyDeprecationWarning,
    stacklevel=2,
)

from ._wcs_helpers import *  # noqa: E402, F401, F403

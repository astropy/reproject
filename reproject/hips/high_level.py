# Licensed under a 3-clause BSD style license - see LICENSE.rst
import warnings

from astropy.utils.exceptions import AstropyDeprecationWarning

warnings.warn(
    "reproject.hips.high_level is a private module and should not be imported "
    "directly. Please use the public API from reproject.hips instead.",
    AstropyDeprecationWarning,
    stacklevel=2,
)

from ._high_level import *  # noqa: E402, F401, F403

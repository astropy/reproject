# Licensed under a 3-clause BSD style license - see LICENSE.rst
import warnings

from astropy.utils.exceptions import AstropyDeprecationWarning

warnings.warn(
    "reproject.adaptive.core is a private module and will be removed in a future "
    "version.  Please use the public API from reproject.adaptive instead.",
    AstropyDeprecationWarning,
    stacklevel=2,
)

from ._core import *  # noqa: E402, F401, F403

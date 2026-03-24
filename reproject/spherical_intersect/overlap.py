# Licensed under a 3-clause BSD style license - see LICENSE.rst
import warnings

from astropy.utils.exceptions import AstropyDeprecationWarning

warnings.warn(
    "reproject.spherical_intersect.overlap is a private module and should not be imported "
    "directly. Please use the public API from reproject.spherical_intersect instead.",
    AstropyDeprecationWarning,
    stacklevel=2,
)

from ._overlap_wrapper import *  # noqa: E402, F401, F403

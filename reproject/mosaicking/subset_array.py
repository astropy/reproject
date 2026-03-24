# Licensed under a 3-clause BSD style license - see LICENSE.rst
import warnings

from astropy.utils.exceptions import AstropyDeprecationWarning

warnings.warn(
    "reproject.mosaicking.subset_array is a private module and should not be imported "
    "directly. Please use the public API from reproject.mosaicking instead.",
    AstropyDeprecationWarning,
    stacklevel=2,
)

from ._subset_array import *  # noqa: E402, F401, F403

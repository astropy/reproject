# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Astropy affiliated package for image reprojection (resampling).
"""

# Affiliated packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from ._astropy_init import *
# ----------------------------------------------------------------------------

if not _ASTROPY_SETUP_:
    from .high_level import reproject
    from .interpolation import reproject_interpolation
    from .spherical_intersect import reproject_flux_conserving
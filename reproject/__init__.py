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
    from .interpolation import reproject_interp
    from .spherical_intersect import reproject_exact
    from .healpix import reproject_from_healpix, reproject_to_healpix

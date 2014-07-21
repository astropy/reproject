# Licensed under a 2-clause BSD style license - see LICENSE.rst

"""
WCS-related utilities
"""
# TODO: The following WCS utilities will likely be merged into Astropy 1.0 and can be
# removed once 0.4 is no longer supported.

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np


__all__ = ['wcs_to_celestial_frame']


def _wcs_to_celestial_frame_builtin(wcs):

    from astropy.coordinates import FK4, FK4NoETerms, FK5, ICRS, Galactic
    from astropy.time import Time
    from astropy.wcs import WCSSUB_CELESTIAL

    # Keep only the celestial part of the axes
    wcs = wcs.sub([WCSSUB_CELESTIAL])

    radesys = wcs.wcs.radesys

    if np.isnan(wcs.wcs.equinox):
        equinox = None
    else:
        equinox = wcs.wcs.equinox

    xcoord = wcs.wcs.ctype[0][:4]
    ycoord = wcs.wcs.ctype[1][:4]

    # Apply logic from FITS standard to determine the default radesys
    if radesys == b'' and xcoord == b'RA--' and ycoord == b'DEC-':
        if equinox is None:
            radesys = "ICRS"
        elif equinox < 1984.:
            radesys = "FK4"
        else:
            radesys = "FK5"

    if radesys == b'FK4':
        if equinox is not None:
            equinox = Time(equinox, format='byear')
        frame = FK4(equinox=equinox)
    elif radesys == b'FK4-NO-E':
        if equinox is not None:
            equinox = Time(equinox, format='byear')
        frame = FK4NoETerms(equinox=equinox)
    elif radesys == b'FK5':
        if equinox is not None:
            equinox = Time(equinox, format='jyear')
        frame = FK5(equinox=equinox)
    elif radesys == b'ICRS':
        frame = ICRS()
    else:
        if xcoord == b'GLON' and ycoord == b'GLAT':
            frame = Galactic()
        else:
            frame = None

    return frame


WCS_FRAME_MAPPINGS = [_wcs_to_celestial_frame_builtin]


def wcs_to_celestial_frame(wcs):
    """WCS to celestial frame.

    TODO: document and test.
    """
    for func in WCS_FRAME_MAPPINGS:
        frame = func(wcs)
        if frame is not None:
            return frame
    raise ValueError("Could not determine celestial frame corresponding "
                     "to the specified WCS object")

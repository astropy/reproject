# Licensed under a 2-clause BSD style license - see LICENSE.rst

"""
WCS-related utilities
"""
# TODO: The following WCS utilities will likely be merged into Astropy 1.0 and can be
# removed once 0.4 is no longer supported.

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np

from astropy import units as u
from astropy.coordinates import UnitSphericalRepresentation


__all__ = ['wcs_to_celestial_frame']


def convert_world_coordinates(xw_in, yw_in, wcs_in, wcs_out):

    # Find input/output frames
    frame_in = wcs_to_celestial_frame(wcs_in)
    frame_out = wcs_to_celestial_frame(wcs_out)

    xw_in_unit = u.Unit(wcs_in.wcs.cunit[0])
    yw_in_unit = u.Unit(wcs_in.wcs.cunit[1])

    data = UnitSphericalRepresentation(xw_in * xw_in_unit,
                                       yw_in * yw_in_unit)

    coords_in = frame_in.realize_frame(data)
    coords_out = coords_in.transform_to(frame_out)

    xw_unit_out = u.Unit(wcs_out.wcs.cunit[0])
    yw_unit_out = u.Unit(wcs_out.wcs.cunit[1])

    xw_out = coords_out.spherical.lon.to(xw_unit_out).value
    yw_out = coords_out.spherical.lat.to(yw_unit_out).value

    return xw_out, yw_out


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
    if radesys == '' and xcoord == 'RA--' and ycoord == 'DEC-':
        if equinox is None:
            radesys = "ICRS"
        elif equinox < 1984.:
            radesys = "FK4"
        else:
            radesys = "FK5"

    if radesys == 'FK4':
        if equinox is not None:
            equinox = Time(equinox, format='byear')
        frame = FK4(equinox=equinox)
    elif radesys == 'FK4-NO-E':
        if equinox is not None:
            equinox = Time(equinox, format='byear')
        frame = FK4NoETerms(equinox=equinox)
    elif radesys == 'FK5':
        if equinox is not None:
            equinox = Time(equinox, format='jyear')
        frame = FK5(equinox=equinox)
    elif radesys == 'ICRS':
        frame = ICRS()
    else:
        if xcoord == 'GLON' and ycoord == 'GLAT':
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

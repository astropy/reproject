# Licensed under a 2-clause BSD style license - see LICENSE.rst

"""
WCS-related utilities
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from astropy import units as u
from astropy.coordinates import UnitSphericalRepresentation
from astropy.wcs.utils import wcs_to_celestial_frame

__all__ = ['convert_world_coordinates']


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


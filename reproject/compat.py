# Licensed under a 3-clause BSD style license - see LICENSE.rst

# Functions backported from other modules for compatibility

from __future__ import absolute_import, division, print_function, unicode_literals

from astropy.wcs import WCS
from astropy.coordinates import BaseRADecFrame, FK4, FK4NoETerms, FK5, ICRS, Galactic

try:

    from astropy.wcs.utils import celestial_frame_to_wcs, custom_frame_to_wcs_mappings

except ImportError:  # Astropy < 3.0

    def _celestial_frame_to_wcs_builtin(frame, projection='TAN'):

        # Create a 2-dimensional WCS
        wcs = WCS(naxis=2)

        if isinstance(frame, BaseRADecFrame):

            xcoord = 'RA--'
            ycoord = 'DEC-'
            if isinstance(frame, ICRS):
                wcs.wcs.radesys = 'ICRS'
            elif isinstance(frame, FK4NoETerms):
                wcs.wcs.radesys = 'FK4-NO-E'
                wcs.wcs.equinox = frame.equinox.byear
            elif isinstance(frame, FK4):
                wcs.wcs.radesys = 'FK4'
                wcs.wcs.equinox = frame.equinox.byear
            elif isinstance(frame, FK5):
                wcs.wcs.radesys = 'FK5'
                wcs.wcs.equinox = frame.equinox.jyear
            else:
                return None
        elif isinstance(frame, Galactic):
            xcoord = 'GLON'
            ycoord = 'GLAT'
        else:
            return None

        wcs.wcs.ctype[0] = xcoord + '-' + projection
        wcs.wcs.ctype[1] = ycoord + '-' + projection

        # Make sure that e.g. LONPOLE and other parameters are set
        # wcs.wcs.set()

        return wcs

    FRAME_WCS_MAPPINGS = [[_celestial_frame_to_wcs_builtin]]

    class custom_frame_to_wcs_mappings(object):
        def __init__(self, mappings=[]):
            if hasattr(mappings, '__call__'):
                mappings = [mappings]
            FRAME_WCS_MAPPINGS.append(mappings)

        def __enter__(self):
            pass

        def __exit__(self, type, value, tb):
            FRAME_WCS_MAPPINGS.pop()

    def celestial_frame_to_wcs(frame, projection='TAN'):
        """
        For a given coordinate frame, return the corresponding WCS object.

        Note that the returned WCS object has only the elements corresponding to
        coordinate frames set (e.g. ctype, equinox, radesys).

        Parameters
        ----------
        frame : :class:`~astropy.coordinates.baseframe.BaseCoordinateFrame` subclass instance
            An instance of a :class:`~astropy.coordinates.baseframe.BaseCoordinateFrame`
            subclass instance for which to find the WCS
        projection : str
            Projection code to use in ctype, if applicable

        Returns
        -------
        wcs : :class:`~astropy.wcs.WCS` instance
            The corresponding WCS object

        Examples
        --------

        ::

            >>> from astropy.wcs.utils import celestial_frame_to_wcs
            >>> from astropy.coordinates import FK5
            >>> frame = FK5(equinox='J2010')
            >>> wcs = celestial_frame_to_wcs(frame)
            >>> wcs.to_header()
            WCSAXES =                    2 / Number of coordinate axes
            CRPIX1  =                  0.0 / Pixel coordinate of reference point
            CRPIX2  =                  0.0 / Pixel coordinate of reference point
            CDELT1  =                  1.0 / [deg] Coordinate increment at reference point
            CDELT2  =                  1.0 / [deg] Coordinate increment at reference point
            CUNIT1  = 'deg'                / Units of coordinate increment and value
            CUNIT2  = 'deg'                / Units of coordinate increment and value
            CTYPE1  = 'RA---TAN'           / Right ascension, gnomonic projection
            CTYPE2  = 'DEC--TAN'           / Declination, gnomonic projection
            CRVAL1  =                  0.0 / [deg] Coordinate value at reference point
            CRVAL2  =                  0.0 / [deg] Coordinate value at reference point
            LONPOLE =                180.0 / [deg] Native longitude of celestial pole
            LATPOLE =                  0.0 / [deg] Native latitude of celestial pole
            RADESYS = 'FK5'                / Equatorial coordinate system
            EQUINOX =               2010.0 / [yr] Equinox of equatorial coordinates


        Notes
        -----

        To extend this function to frames not defined in astropy.coordinates, you
        can write your own function which should take a
        :class:`~astropy.coordinates.baseframe.BaseCoordinateFrame` subclass
        instance and a projection (given as a string) and should return either a WCS
        instance, or `None` if the WCS could not be determined. You can register
        this function temporarily with::

            >>> from astropy.wcs.utils import celestial_frame_to_wcs, custom_frame_to_wcs_mappings
            >>> with custom_frame_to_wcs_mappings(my_function):
            ...     celestial_frame_to_wcs(...)

        """
        for mapping_set in FRAME_WCS_MAPPINGS:
            for func in mapping_set:
                wcs = func(frame, projection=projection)
                if wcs is not None:
                    return wcs
        raise ValueError("Could not determine WCS corresponding to the specified "
                         "coordinate frame.")

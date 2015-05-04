**************
HEALPIX images
**************

Images can also be stored using the HEALPIX representation, and the
*reproject* package includes two functions,
:func:`~reproject.reproject_from_healpix` and
:func:`~reproject.reproject_to_healpix`, which can be used to reproject
from/to HEALPIX representations (these functions are wrappers around
functionality provided by the `healpy <http://healpy.readthedocs.org>`_
package). The functions can be imported with::

    >>> from reproject import reproject_from_healpix, reproject_to_healpix

The :func:`~reproject.reproject_from_healpix` function takes either a
filename, a FITS Table HDU object, or a tuple containing a 1-D array and a
coordinate frame given as an Astropy :class:`~astropy.coordinates.BaseCoordinateFrame`
instance or a string. The target
projection should be given either as a WCS object (which required you to also
specify the output shape using ``shape_out``) or as a FITS
:class:`~astropy.io.fits.Header` object. All of the following are valid ways
of reprojecting HEALPIX data::

    >>> array, footprint = reproject_from_healpix('my_healpix_map.fits', target_header)
    >>> array, footprint = reproject_from_healpix('my_healpix_map.fits', target_wcs, shape_out=(100,100))
    >>> array, footprint = reproject_from_healpix((data, 'fk5'), target_header)
    >>> array, footprint = reproject_from_healpix((data, FK5(equinox='J2010')), target_header)
    
On the other hand, the :func:`~reproject.reproject_to_healpix` function takes
input data in the same form as :func:`~reproject.reproject_interpolation`
(see :ref:`interpolation`) for the first argument, and a coordinate frame as the
second argument, either as a string or as a
:class:`~astropy.coordinates.BaseCoordinateFrame` instance e.g.::

    >>> array, footprint = reproject_to_healpix((array, header_in), 'galactic')

"""HEALPIX (Hierarchical Equal-Area and Isolatitude Pixelization) utility functions.

This is a thin wrapper convenience functions around
`healpy` (http://code.google.com/p/healpy/) functionality.

Refer to https://github.com/healpy/healpy/issues/129 and https://github.com/gammapy/gammapy/blob/master/gammapy/image/healpix.py
"""
from __future__ import print_function, division
from astropy import wcs
from astropy.io import fits
import numpy as np

__all__ = ['healpix_to_image', 'image_to_healpix']

valid_coordinate_systems = ('galactic','icrs')

def healpix_reproject_file(hp_filename, reference, outfilename=None, clobber=False, ext=1, **kwargs):
    """
    Reproject a HEALPIX file

    Parameters
    ----------
    hp_filename : str
        A HEALPIX FITS file name
    reference : fits.Header, fits.PrimaryHDU, fits.HDUList, or str
        A fits.Header or HDU or FITS filename containing the target for projection
    outfilename : str or None
        The filename to write to
    clobber : bool
        Overwrite the outfilename if it exists?
    ext : int
        The FITS extension containing the HEALPIX table to reproject
    kwargs : dict
        passed to healpix_hdu_to_hdu

    Returns
    -------
    fits.PrimaryHDU containing the reprojected image
    """
    hp_hdu = fits.open(hp_filename)[ext]

    if isinstance(reference,str):
        reference_header = fits.getheader(reference)
    elif isinstance(reference,fits.Header):
        reference_header = reference
    elif isinstance(reference,fits.PrimaryHDU):
        reference_header = reference.header
    elif isinstance(reference,fits.HDUList):
        reference_header = reference[0].header
    else:
        raise TypeError("Reference was not a valid type; must be some sort of FITS header representation")

    new_hdu = healpix_hdu_to_hdu(hp_hdu, reference_header, **kwargs)

    if outfilename is not None:
        new_hdu.writeto(outfilename, clobber=clobber)

    return new_hdu

def healpix_hdu_to_hdu(hp_hdu, reference_header, field=None, **kwargs):
    """
    Convert a HEALPIX binary-table HDU to a target FITS header.

    Parameters
    ----------
    hp_hdu : fits.BinTableHDU
        The HDU containing the healpix data table
    reference_header : fits.Header
        The target header, containing a valid WCS
    field : None or string
        The field name containing the data to be reprojected.  If not
        specifies, defaults to the first field in the BinTable
        
    Returns
    -------
    fits.PrimaryHDU containing the reprojected image
    """
    if field is None:
        field = hp_hdu.data.columns[0].name

    nested = hp_hdu.header['ORDERING'] == 'NESTED'
    coordsys = hp_hdu.header['COORDSYS'].lower()

    if coordsys not in valid_coordinate_systems:
        raise KeyError("Coordinate system was %s, which is not one of %s" % (coordsys,valid_coordinate_systems))

    healpix_data = hp_hdu.data[field]

    reprojected_data = healpix_to_image(healpix_data, reference_header,
                                        coordsys, nest=nested, **kwargs)

    new_hdu = fits.PrimaryHDU(data=reprojected_data, header=reference_header)

    return new_hdu

def healpix_to_image(healpix_data, reference_header, hpx_coord_system,
                     interp=True, nest=False):
    """
    Convert image in HEALPIX format to a normal FITS projection image (e.g.
    CAR or AIT).

    Parameters
    ----------
    healpix_data : `numpy.ndarray`
        HEALPIX data array
    reference_image : `astropy.io.fits.ImageHDU`
        A reference image to project to.  Must have a 'COORDSYS' keyword of
        either 'galactic' or 'icrs'
    hpx_coord_system : 'galactic' or 'icrs'
        The target coordinate system.  Should be derived from the HEALPIX
        COORDSYS keyword if it is a FITS file
    nest : bool
        The order of the healpix_data, either nested or ring.  Stored in 
        FITS headers in the ORDERING keyword.
    interp : bool
        Get the bilinear interpolated data?  If not, returns a set of neighbors

    Returns
    -------
    reprojected_data : `numpy.ndarray`
        HEALPIX image resampled onto the reference image
    
    Examples
    --------
    >>> import healpy as hp
    >>> from astropy.io import fits
    >>> from reproject.healpix import healpix_to_image
    >>> healpix_filename = 'healpix.fits'
    >>> healpix_data = hp.read_map(healpix_filename)
    >>> healpix_system = fits.getheader(healpix_filename,ext=1)['COORDSYS']
    >>> healpix_isnested = fits.getheader(healpix_filename,ext=1)['ORDERING'] == 'NESTED'
    >>> reference_image = fits.open('reference_image.fits')[0]
    >>> reprojected_data = healpix_to_image(healpix_data, reference_image, healpix_system, nest=healpix_isnested)
    >>> fits.writeto('new_image.fits', reprojected_data, reference_image.header)
    
    >>> neighbors = healpix_to_image(healpix_data, reference_image, healpix_system, nest=healpix_isnested, interp=False)
    >>> reprojected_data = healpix_data[neighbors].mean(axis=0)
    >>> fits.writeto('new_image.fits', reprojected_data, reference_image.header)
    """
    import healpy as hp
    refwcs = wcs.WCS(reference_header)
    yinds,xinds = np.indices([reference_header['NAXIS2'],reference_header['NAXIS1']])
    lon_deg, lat_deg = refwcs.wcs_pix2world(xinds,yinds,0)
    lon, lat = np.radians(lon_deg), np.radians(lat_deg)
    
    # If the reference image uses a different celestial coordinate system from
    # the HEALPIX image we need to transform the coordinates
    ref_coord_system = reference_header['COORDSYS']
    if ref_coord_system != hpx_coord_system:
        from ..utils.coordinates import sky_to_sky
        lon, lat = sky_to_sky(lon, lat, ref_coord_system, hpx_coord_system)
    
    # theta must be in the range [0,pi], but wcs gives it in the range (-pi/2,pi/2)
    # also, somehow, lat=pi/2 corresponds to the SOUTH galactic pole and lat=-pi/2 the NORTH
    # (this confuses me, but test it yourself by comparing some Galactic
    # surveys... W51, M17, and others are flipped if you don't use -lat)
    if interp:
        data = hp.get_interp_val(healpix_data, theta=-lat+np.pi/2, phi=lon, nest=nest)
    else:
        neighbors = hp.get_all_neighbours(2048, theta=-lat+np.pi/2, phi=lon, nest=nest)
        data = healpix_data[neighbors]

    return data


def sky_to_sky(lon, lat, in_system, out_system, unit='deg'):
    """Convert between sky coordinates.
    (utility function - likely belongs somewhere else)
    copied from gammapy: https://github.com/gammapy/gammapy/blob/master/gammapy/utils/coordinates/celestial.py#L189

    Parameters
    ----------
    lon, lat : array_like
        Coordinate arrays
    in_system, out_system : {'galactic', 'icrs'}
        Input / output coordinate system
    unit : {'deg', 'rad'}
        Units of input and output coordinates

    Returns
    -------
    """    
    from astropy.coordinates import ICRS, Galactic
    systems = dict(galactic=Galactic, icrs=ICRS)

    lon = np.asanyarray(lon)
    lat = np.asanyarray(lat)

    in_coords = systems[in_system](lon, lat, unit=(unit, unit))
    out_coords = in_coords.transform_to(systems[out_system])
    
    if unit == 'deg':
        return out_coords.lonangle.deg, out_coords.latangle.deg
    else:
        return out_coords.lonangle.rad, out_coords.latangle.rad


def image_to_healpix(image, healpix_pars):
    """Convert image in a normal FITS projection (e.g. CAR or AIT) to HEALPIX format. 

    Parameters
    ----------
    image : `astropy.io.fits.ImageHDU`
        The input image
    healpix_pars : TODO
        TODO: what HEALPIX parameters do we need?
    Returns
    -------
    healpix_data : `numpy.array`
        HEALPIX array data
    """
    raise NotImplementedError
    # Can we use Kapteyn or Healpy to get e.g. bilinear interpolation?


import numpy as np

from astropy.io import fits
from astropy.extern import six
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel, pixel_to_skycoord

__all__ = ['reproject_events', 'parse_events']


def parse_events(hdu, weights=None):
    """
    Parse events from a FITS file or HDU.

    Parameters
    ----------
    hdu : str or `~astropy.io.fits.BinTableHDU` or `~astropy.io.fits.TableHDU`
        The FITS file or HDU to extract events from.
    weights : str, optional
        If specified, this column in the event list will be used to extract the
        weights.

    Returns
    -------
    coords : `~astropy.coordinates.SkyCoord`
        The celestial coordinates of the events
    weights : `~numpy.ndarray`
        The weights for each event
    """

    if isinstance(hdu, six.string_types):
        with fits.open(hdu) as hdulist:
            try:
                hdu = hdulist['EVENTS']
            except KeyError:
                raise ValueError("Could not find an EVENTS HDU. Try passing the HDU object directly instead of the filename.")
            return parse_events(hdu)
    elif not isinstance(hdu, (fits.TableHDU, fits.BinTableHDU)):
        raise TypeError("hdu should be a filename or a table HDU instance")

    colnames = [col.name.lower() for col in hdu.columns]

    x_idx = colnames.index('x') + 1
    y_idx = colnames.index('y') + 1

    wcs = WCS(hdu.header, keysel=['pixel'], colsel=(x_idx, y_idx))

    xp = hdu.data['x']
    yp = hdu.data['y']

    coords = pixel_to_skycoord(xp, yp, wcs, origin=1)

    if weights is not None:
        weights = hdu.data[weights]
    else:
        weights = np.ones(len(hdu.data))

    return coords, weights


def reproject_events(events, wcs_out, shape_out):
    """
    Reproject events onto an output WCS.

    Parameters
    ----------
    events : tuple
        A tuple of celestial coordinates as a `~astropy.coordinates.SkyCoord`
        object and weights as an `~numpy.ndarray`.
    wcs_out : `~astropy.wcs.WCS`
        The output WCS
    shape_out : tuple
        The shape of the output array

    Returns
    -------
    array_new : `~numpy.ndarray`
        The reprojected events
    footprint : None
        The footprint is always set to `None`
    """

    coords, weights = events
    ny, nx = shape_out

    # coords is a SkyCoord which we can convert to pixel coordinates
    xp, yp = skycoord_to_pixel(coords, wcs_out, origin=0)

    # we now bin values into histogram
    hist = np.histogram2d(yp, xp, weights=weights, bins=shape_out, range=[[-0.5, ny - 0.5], [-0.5, nx - 0.5]])[0]

    return hist, None

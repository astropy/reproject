import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from ._overlap import _compute_overlap

def reproject(hdu_in, header_out):
    """
    Reproject an image given by a FITS HDU onto a new Header

    Parameters
    ----------
    hdu_in : `~astropy.io.fits.PrimaryHDU` or `~astropy.io.fits.ImageHDU`
        The HDU containing the image and original header
    header_out : `~astropy.io.fits.Header`
        The new header to project to

    Returns
    -------
    hdu_out: `~astropy.io.fits.ImageHDU`
    """

    if not isinstance(hdu_in, (fits.ImageHDU, fits.PrimaryHDU)):
        raise TypeError("hdu_in should be an ImageHDU instance")

    if not isinstance(header_out, fits.Header):
        raise TypeError("header_out should be a Header instance")

    # Parse input WCS
    wcs_in = WCS(hdu_in.header)

    # Parse output WCS
    wcs_out = WCS(header_out)

    # Start off by finding the world position of all the corners of the input
    # image in world coordinates

    x = np.arange(hdu_in.header['NAXIS1'] + 1.) - 0.5
    y = np.arange(hdu_in.header['NAXIS2'] + 1.) - 0.5

    x_pix_in, y_pix_in = np.meshgrid(x, y)

    x_world_in, y_world_in = wcs_in.wcs_pix2world(x_pix_in, y_pix_in, 0)

    # Now compute the world positions of all the corners in the output header

    x = np.arange(header_out['NAXIS1'] + 1.) - 0.5
    y = np.arange(header_out['NAXIS2'] + 1.) - 0.5

    x_pix_out, y_pix_out = np.meshgrid(x, y)

    x_world_out, y_world_out = wcs_out.wcs_pix2world(x_pix_out, y_pix_out, 0)

    # Finally, compute the pixel positions in the *output* image of the pixels
    # from the *input* image.

    x_pix_inout, y_pix_inout = wcs_out.wcs_world2pix(x_world_in, y_world_in, 0)

    # Create output image

    hdu_out = fits.PrimaryHDU(header=header_out)
    hdu_out.data = np.zeros((header_out['NAXIS2'], header_out['NAXIS1']))
    weights = np.zeros(hdu_out.data.shape)

    for i in range(hdu_in.header['NAXIS1']):
        for j in range(hdu_in.header['NAXIS2']):

            # For every input pixel we find the position in the output image in
            # pixel coordinates, then use the full range of overlapping output
            # pixels with the exact overlap function.

            xmin = int(min(x_pix_inout[j, i], x_pix_inout[j, i+1], x_pix_inout[j+1, i+1], x_pix_inout[j+1, i]))
            xmax = int(max(x_pix_inout[j, i], x_pix_inout[j, i+1], x_pix_inout[j+1, i+1], x_pix_inout[j+1, i]))
            ymin = int(min(y_pix_inout[j, i], y_pix_inout[j, i+1], y_pix_inout[j+1, i+1], y_pix_inout[j+1, i]))
            ymax = int(max(y_pix_inout[j, i], y_pix_inout[j, i+1], y_pix_inout[j+1, i+1], y_pix_inout[j+1, i]))

            ilon = [[x_world_in[j, i], x_world_in[j, i+1], x_world_in[j+1, i+1], x_world_in[j+1, i]][::-1]]
            ilat = [[y_world_in[j, i], y_world_in[j, i+1], y_world_in[j+1, i+1], y_world_in[j+1, i]][::-1]]
            ilon = np.radians(np.array(ilon))
            ilat = np.radians(np.array(ilat))

            xmin = max(0, xmin)
            xmax = min(hdu_out.header["NAXIS1"]-1, xmax)
            ymin = max(0, ymin)
            ymax = min(hdu_out.header["NAXIS2"]-1, ymax)

            for ii in range(xmin, xmax+1):
                for jj in range(ymin, ymax+1):


                    olon = [[x_world_out[jj, ii], x_world_out[jj, ii+1], x_world_out[jj+1, ii+1], x_world_out[jj+1, ii]][::-1]]
                    olat = [[y_world_out[jj, ii], y_world_out[jj, ii+1], y_world_out[jj+1, ii+1], y_world_out[jj+1, ii]][::-1]]
                    olon = np.radians(np.array(olon))
                    olat = np.radians(np.array(olat))

                    # Figure out the fraction of the input pixel that makes it
                    # to the output pixel at this position.

                    overlap, _ = _compute_overlap(ilon, ilat, olon, olat)
                    original, _ = _compute_overlap(ilon, ilat, ilon, ilat)
                    hdu_out.data[ii, jj] += hdu_in.data[i, j] * overlap / original
                    weights[ii,jj] += overlap / original

    hdu_out.data /= weights

    return hdu_out

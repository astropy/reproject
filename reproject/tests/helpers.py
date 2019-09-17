from astropy.io import fits


def array_footprint_to_hdulist(array, footprint, header):
    hdulist = fits.HDUList()
    hdulist.append(fits.PrimaryHDU(array, header))
    hdulist.append(fits.ImageHDU(footprint, header, name='footprint'))
    return hdulist

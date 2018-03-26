import numpy as np
import sys
from astropy.io import fits
from astropy.io.fits import PrimaryHDU, ImageHDU, CompImageHDU, Header, HDUList
from astropy.wcs import WCS
from astropy.extern import six


def parse_input_data(input_data, hdu_in=None):
    """
    Parse input data to return a Numpy array and WCS object.
    """

    if isinstance(input_data, six.string_types):
        return parse_input_data(fits.open(input_data), hdu_in=hdu_in)
    elif isinstance(input_data, HDUList):
        if hdu_in is None:
            if len(input_data) > 1:
                raise ValueError("More than one HDU is present, please specify HDU to use with ``hdu_in=`` option")
            else:
                hdu_in = 0
        return parse_input_data(input_data[hdu_in])
    elif isinstance(input_data, (PrimaryHDU, ImageHDU, CompImageHDU)):
        return input_data.data, WCS(input_data.header)
    elif isinstance(input_data, tuple) and isinstance(input_data[0], np.ndarray):
        if isinstance(input_data[1], Header):
            return input_data[0], WCS(input_data[1])
        else:
            return input_data
    else:
        raise TypeError("input_data should either be an HDU object or a tuple of (array, WCS) or (array, Header)")


def parse_output_projection(output_projection, shape_out=None):
    string_type = str if sys.version_info[0] == 3 else basestring
    if isinstance(output_projection, Header):
        wcs_out = WCS(output_projection)
        try:
            shape_out = [output_projection['NAXIS{0}'.format(i + 1)] for i in range(output_projection['NAXIS'])][::-1]
        except KeyError:
            if shape_out is None:
                raise ValueError("Need to specify shape since output header does not contain complete shape information")
    elif isinstance(output_projection, WCS):
        wcs_out = output_projection
        if shape_out is None:
            raise ValueError("Need to specify shape when specifying output_projection as WCS object")
    elif isinstance(output_projection, string_type):
        print(string_type)
        hdu_list = fits.open(output_projection)
        shape_out = hdu_list[0].data.shape
        header = hdu_list[0].header
        wcs_out = WCS(header)
    else:
        raise TypeError('output_projection should either be a Header, a WCS object, or a filename')

    if len(shape_out) == 0:
        raise ValueError("The shape of the output header should not be an empty tuple")
    return wcs_out, shape_out

import numpy as np

from astropy.io.fits import PrimaryHDU, ImageHDU, CompImageHDU, Header
from astropy.wcs import WCS

def parse_input_data(input_data):
    """
    Parse input data to return a Numpy array and WCS object.
    """
    
    if isinstance(input_data, (PrimaryHDU, ImageHDU, CompImageHDU)):
        array_in = input_data.data
        wcs_in = WCS(input_data.header)
    elif isinstance(input_data, tuple) and isinstance(input_data[0], np.ndarray):
        array_in = input_data[0]
        if isinstance(input_data[1], Header):
            wcs_in = WCS(input_data[1])
        else:
            wcs_in = input_data[1]
    else:
        raise TypeError("input_data should either be an HDU object or a tuple of (array, WCS) or (array, Header)")

    return array_in, wcs_in

def parse_output_projection(output_projection, shape_out=None):
    
    if isinstance(output_projection, Header):
        wcs_out = WCS(output_projection)
        try:
            shape_out = [output_projection['NAXIS{0}'.format(i+1)] for i in range(output_projection['NAXIS'])][::-1]
        except KeyError:
            if shape_out is None:
                raise ValueError("Need to specify shape since output header does not contain complete shape information")
    elif isinstance(output_projection, WCS):
        wcs_out = output_projection
        if shape_out is None:
            raise ValueError("Need to specify shape when specifying output_projection as WCS object")

    return wcs_out, shape_out
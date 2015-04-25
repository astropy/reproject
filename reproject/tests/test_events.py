import os

import numpy as np

from io import StringIO
from astropy.io import fits

from ..events import parse_events
from ..high_level import reproject

HEADER = fits.Header.fromtextfile(StringIO("""
SIMPLE  = T
BITPIX  = -64
NAXIS   = 2
NAXIS1  = 22
NAXIS2  = 23
CRPIX1  = 105
CRVAL1  = 333.
CDELT1  = -0.003
CTYPE1  = 'RA---TAN'
CRPIX2  = 75.
CRVAL2  = -22
CDELT2  = 0.003
CTYPE2  = 'DEC--TAN'
CROTA2  = 0.000000000
END
"""))

DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))


def test_events_filename():
    
    coords, weights = parse_events(os.path.join(DATA, 'events.fits'))
    
    result, footprint = reproject((coords, weights), HEADER)
    
    reference = fits.getdata(os.path.join(DATA, 'reference.fits'))
    
    np.testing.assert_allclose(reference, result)
    assert footprint is None
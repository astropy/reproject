"""Example how to reproject by interpolation.
"""
import numpy as np
from reproject.interpolation import reproject_celestial_slices
from astropy.io import fits
from astropy.wcs import WCS
from wcsaxes import datasets

# Test 2d interpolation, different frame, different projection

hdu = datasets.msx_hdu()
hdu.data[100:200, 100:200] = np.inf
wcs_in = WCS(hdu.header)
wcs_out = wcs_in.deepcopy()
wcs_out.wcs.ctype = ['RA---TAN', 'DEC--TAN']
wcs_out.wcs.crval = [266.44707, -28.937888]

array_out = reproject_celestial_slices(hdu.data, wcs_in, wcs_out, hdu.data.shape)

fits.writeto('test_2d.fits', array_out,
              header=wcs_out.to_header(), clobber=True)

# Test 3d slice-by-slice interpolation, different frame, different projection

hdu = datasets.l1448_co_hdu()
wcs_in = WCS(hdu.header)
wcs_in.wcs.equinox = 2000.
wcs_out = wcs_in.deepcopy()
wcs_out.wcs.ctype = ['GLON-SIN', 'GLAT-SIN', wcs_in.wcs.ctype[2]]
wcs_out.wcs.crval = [158.0501, -21.530282, wcs_in.wcs.crval[2]]
wcs_out.wcs.crpix = [50., 50., wcs_in.wcs.crpix[2]]

array_out = reproject_celestial_slices(hdu.data, wcs_in, wcs_out, hdu.data.shape)

fits.writeto('test_3d.fits', array_out,
            header=wcs_out.to_header(), clobber=True)

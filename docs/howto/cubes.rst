.. _reprojecting-cubes:

****************************
Reprojecting a spectral cube
****************************

The :func:`~reproject.reproject_interp` function is not limited to
2-dimensional images - it can reproject data with any number of dimensions,
as long as the input and output WCS have the same number of dimensions as the
data. For a spectral cube, this means that the spectral axis is resampled
along with the celestial axes in a single call.

As an example, we can set up a small synthetic cube with two celestial axes
and one spectral axis:

    >>> import numpy as np
    >>> from astropy.wcs import WCS
    >>> cube = np.ones((24, 30, 30))
    >>> wcs_in = WCS(naxis=3)
    >>> wcs_in.wcs.ctype = 'RA---TAN', 'DEC--TAN', 'VELO-LSR'
    >>> wcs_in.wcs.crpix = 15.5, 15.5, 1
    >>> wcs_in.wcs.crval = 40., 0., 0.
    >>> wcs_in.wcs.cdelt = -0.01, 0.01, 500.

We can then reproject this to an output WCS which differs both in the
celestial axes (a different projection) and in the spectral axis (channels
twice as wide):

    >>> from reproject import reproject_interp
    >>> wcs_out = wcs_in.deepcopy()
    >>> wcs_out.wcs.ctype = 'RA---CAR', 'DEC--CAR', 'VELO-LSR'
    >>> wcs_out.wcs.cdelt = -0.01, 0.01, 1000.
    >>> new_cube, footprint = reproject_interp((cube, wcs_in), wcs_out,
    ...                                        shape_out=(12, 30, 30))
    >>> new_cube.shape
    (12, 30, 30)

As for images, the input can also be given as e.g. a FITS filename or an HDU
object - see :ref:`input-formats` for the full list of supported inputs.

Note that full n-dimensional reprojection is only available for
:func:`~reproject.reproject_interp` - the
:func:`~reproject.reproject_adaptive` and :func:`~reproject.reproject_exact`
functions can only reproject the two celestial dimensions (see
:ref:`choosing-algorithm`).

Reprojecting only the celestial axes
====================================

In many cases, the spectral axis of the output should stay the same as that
of the input, and only the celestial axes need to be reprojected. While the
approach above will work (using an output WCS with an identical spectral
axis), it is more efficient to reproject each spectral slice with the same
celestial coordinate transformation, computed once. This works with any of
the reprojection algorithms, including :func:`~reproject.reproject_adaptive`
and :func:`~reproject.reproject_exact`, and is described in detail in
:ref:`broadcasting`.

Large cubes
===========

Spectral cubes can be much larger than memory, in which case reprojecting the
whole cube in one go as above is not an option. See :doc:`chunked` for
carrying out the reprojection in chunks (optionally in parallel),
:doc:`performance` for writing the output to a memory-mapped array, and
:doc:`dask` for working with dask arrays.

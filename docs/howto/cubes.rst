.. _reprojecting-cubes:

****************************
Reprojecting a spectral cube
****************************

The :func:`~reproject.reproject_interp` function is not limited to
2-dimensional images - it can reproject data with any number of dimensions,
as long as the input and output WCS have the same number of dimensions as the
data. For a spectral cube, this means that the spectral axis is resampled
along with the celestial axes in a single call.

As an example, we can download a 13CO spectral cube of the L1448 region from
`http://data.astropy.org <http://data.astropy.org>`_, which has two celestial
axes and one spectral (velocity) axis:

    >>> from astropy.io import fits
    >>> from astropy.utils.data import get_pkg_data_filename
    >>> hdu = fits.open(get_pkg_data_filename('l1448/l1448_13co.fits'))[0]   # doctest: +REMOTE_DATA
    >>> hdu.data.shape   # doctest: +REMOTE_DATA
    (53, 105, 105)

We can then reproject this to an output WCS with spectral channels twice as
wide, halving the number of channels (dividing the reference pixel position
along the spectral axis by two so that the cube covers the same velocity
range):

    >>> from astropy.wcs import WCS
    >>> from reproject import reproject_interp
    >>> wcs_in = WCS(hdu.header)   # doctest: +REMOTE_DATA
    >>> wcs_out = wcs_in.deepcopy()   # doctest: +REMOTE_DATA
    >>> wcs_out.wcs.cdelt[2] = 2 * wcs_in.wcs.cdelt[2]   # doctest: +REMOTE_DATA
    >>> wcs_out.wcs.crpix[2] = (wcs_in.wcs.crpix[2] + 1) / 2   # doctest: +REMOTE_DATA
    >>> new_cube, footprint = reproject_interp(hdu, wcs_out,
    ...                                        shape_out=(27, 105, 105))   # doctest: +REMOTE_DATA
    >>> new_cube.shape   # doctest: +REMOTE_DATA
    (27, 105, 105)

In this example only the spectral axis changes, but the celestial axes can be
changed in the same call, for example to use a different projection or
resolution. As for images, the input can also be given as e.g. a FITS
filename or a plain array with a WCS - see :ref:`input-formats` for the full
list of supported inputs.

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

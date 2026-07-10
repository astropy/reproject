.. _celestial:

**********************************
Regular celestial images and cubes
**********************************


.. _common:

Common options
==============

All of the reprojection algorithms implemented in *reproject* are available
as functions named as ``reproject_<algorithm>``, e.g.
:func:`~reproject.reproject_interp`, :func:`~reproject.reproject_adaptive`,
and :func:`~reproject.reproject_exact`. These can be imported from the top-level
of the package, e.g.::

    >>> from reproject import reproject_interp
    >>> from reproject import reproject_adaptive
    >>> from reproject import reproject_exact

All functions share a common set of arguments, as well as including
algorithm-dependent arguments. In this section, we take a look at some of the
common arguments.

The reprojection functions take two main arguments. The first argument is the
image to reproject, together with WCS information about the image. This can be
either:

* The name of a FITS file
* An :class:`~astropy.io.fits.HDUList` object
* An image HDU object such as a :class:`~astropy.io.fits.PrimaryHDU`,
  :class:`~astropy.io.fits.ImageHDU`, or
  :class:`~astropy.io.fits.CompImageHDU` instance
* A tuple where the first element is a :class:`~numpy.ndarray` and the
  second element is either a :class:`~astropy.wcs.WCS` or a
  :class:`~astropy.io.fits.Header` object

In the case of a FITS file or an :class:`~astropy.io.fits.HDUList` object, if
there is more than one Header-Data Unit (HDU), the ``hdu_in`` keyword argument
is also required and should be set to the ID or the name of the HDU to use.

The second argument is the WCS information for the output image, which should be
specified either as a :class:`~astropy.wcs.WCS` or a
:class:`~astropy.io.fits.Header` instance. If this is specified as a
:class:`~astropy.wcs.WCS` instance, the ``shape_out`` keyword argument should
also be specified, and be given the shape of the output image using the Numpy
``(ny, nx)`` convention (this is because :class:`~astropy.wcs.WCS`, unlike
:class:`~astropy.io.fits.Header`, does not contain information about image
size).

As an example, we start off by opening a FITS file using Astropy::

    >>> from astropy.io import fits
    >>> hdu = fits.open('http://data.astropy.org/galactic_center/gc_msx_e.fits')[0]    # doctest: +REMOTE_DATA +IGNORE_OUTPUT
    Downloading http://data.astropy.org/galactic_center/gc_msx_e.fits [Done]

The image is currently using a Plate Carée projection::

    >>> hdu.header['CTYPE1']   # doctest: +REMOTE_DATA
    'GLON-CAR'

We can create a new header using a Gnomonic projection::

    >>> new_header = hdu.header.copy()   # doctest: +REMOTE_DATA
    >>> new_header['CTYPE1'] = 'GLON-TAN'   # doctest: +REMOTE_DATA
    >>> new_header['CTYPE2'] = 'GLAT-TAN'   # doctest: +REMOTE_DATA

And finally we can call the :func:`~reproject.reproject_interp` function to reproject
the image using interpolation::

    >>> from reproject import reproject_interp
    >>> new_image, footprint = reproject_interp(hdu, new_header)   # doctest: +REMOTE_DATA

The reprojection functions return two arrays - the first is the reprojected
input image, and the second is a 'footprint' array which shows the fraction of
overlap of the input image on the output image grid. This footprint is 0 for
output pixels that fall outside the input image, 1 for output pixels that fall
inside the input image. For more information about footprint arrays, see the
:ref:`footprints` section. To return only the main array and not the footprint,
you can set ``return_footprint=False``.

We can then easily write out the reprojected image to a new FITS file::

    >>> fits.writeto('reprojected_image.fits', new_image, new_header)   # doctest: +REMOTE_DATA

.. _interpolation:

Interpolation
=============

The :func:`~reproject.reproject_interp` function can be used to carry out
reprojection implemented using simple interpolation::

    >>> from reproject import reproject_interp

In addition to the arguments described in :ref:`common`, the order of the
interpolation can be controlled by setting the ``order=`` argument to either an
integer or a string giving the order of the interpolation. Supported strings
include:

* ``'nearest-neighbor'``: zeroth order interpolation
* ``'bilinear'``: first order interpolation
* ``'biquadratic'``: second order interpolation
* ``'bicubic'``: third order interpolation

.. _adaptive:

Adaptive resampling
===================

The :func:`~reproject.reproject_adaptive` function can be used to carry out
anti-aliased reprojection using the  `DeForest (2004)
<https://doi.org/10.1023/B:SOLA.0000021743.24248.b0>`_ algorithm::

    >>> from reproject import reproject_adaptive

This algorithm provides high-quality photometry through anti-aliased
reprojection, which avoids the problems of plain interpolation when the input
and output images have different resolutions, and it offers a flux-conserving
mode.

.. _exact:

Spherical Polygon Intersection
==============================

The :func:`~reproject.reproject_exact` function can be used to carry out 'exact'
reprojection using the spherical polygon intersection of input and output pixels::

    >>> from reproject import reproject_exact

For this algorithm, the footprint array returned gives the exact fractional
overlap of new pixels with the original image (see :ref:`footprints` for more
details).

.. warning:: The :func:`~reproject.reproject_exact` is currently known to
             have precision issues for images with resolutions <0.05". For
             now it is therefore best to avoid using it with such images.

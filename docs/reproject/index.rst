*********************************************
Image reprojection (resampling) (`reproject`)
*********************************************

Introduction
============

The `reproject` package implements image reprojection (resampling) methods
for astronomical images.


.. note::

    We plan to propose that `reproject` will be merged into the
    ``astropy`` core as ``astropy.reproject`` once the main functionality
    is in place and has been tested for a while.

.. note::

    `reproject` requires `numpy <http://www.numpy.org/>`__ and
    `astropy <http://www.astropy.org/>`__ >=1.0 to be installed.
    Some functionality is only available if `scipy <http://www.scipy.org/>`__ or
    `scikit-image <http://scikit-image.org/>`__ are installed, users are
    encouraged to install those optional dependencies.

Getting Started
===============

The easiest way to reproject an image is to make use of the high-level
:func:`~reproject.reproject` function::

    >>> from reproject import reproject

This function takes two main arguments. The first argument is the image to
reproject, together with WCS information about the image. This can be either an
Astropy HDU object (specifically :class:`~astropy.io.fits.PrimaryHDU` or
:class:`~astropy.io.fits.ImageHDU`), or a tuple with two elements: a Numpy
array and either a :class:`~astropy.wcs.WCS` or a
:class:`~astropy.io.fits.Header` instance.

The second argument is the WCS information for the output image, which should
be specified either as a :class:`~astropy.wcs.WCS` or a
:class:`~astropy.io.fits.Header` instance. If this is specified as a
:class:`~astropy.wcs.WCS` instance, the ``shape_out`` argument to
:func:`~reproject.reproject` should also be specified, and give the shape of
the output image using the Numpy ``(ny, nx)`` convention (this is because
:class:`~astropy.wcs.WCS`, unlike :class:`~astropy.io.fits.Header`, does not
contain information about image size).

We start off by opening a FITS file using Astropy::

    >>> from astropy.io import fits
    >>> hdu = fits.open('https://github.com/aplpy/aplpy-examples/raw/master/data/MSX_E.fits.gz')[0]   # doctest: +REMOTE_DATA
    Downloading https://github.com/aplpy/aplpy-examples/raw/master/data/MSX_E.fits.gz [Done]

The image is currently using a Plate Carée projection::

    >>> hdu.header['CTYPE1']   # doctest: +REMOTE_DATA
    'GLON-CAR'

We can create a new header using a Gnomonic projection::

    >>> new_header = hdu.header.copy()   # doctest: +REMOTE_DATA
    >>> new_header['CTYPE1'] = 'GLON-TAN'   # doctest: +REMOTE_DATA
    >>> new_header['CTYPE2'] = 'GLAT-TAN'   # doctest: +REMOTE_DATA

And finally we can call the :func:`~reproject.reproject` function to reproject
the image::

    >>> from reproject import reproject
    >>> new_image, footprint = reproject(hdu, new_header)   # doctest: +REMOTE_DATA

The :func:`~reproject.reproject` function returns two arrays - the first is the
reprojected input image, and the second is a 'footprint' array which shows the
fraction of overlap of the input image on the output image grid. This footprint
is 0 for output pixels that fall outside the input image, 1 for output pixels
that fall completely inside the input image, and values between 0 and 1 for
pixels with partial overlap.

We can then easily write out the reprojected image to a new FITS file::

    >>> fits.writeto('reprojected_image.fits', new_image, new_header)   # doctest: +REMOTE_DATA

There are different reprojection methods implemented. By default, the
reprojection is done using bilinear interpolation, which is very fast but not
flux-conserving. The reprojection method can be explicitly set with the
``projection_type`` argument, which can be one of:

* ``'nearest-neighbor'``: zeroth order interpolation
* ``'bilinear'``: fisst order interpolation
* ``'biquadratic'``: second order interpolation
* ``'bicubic'``: third order interpolation
* ``'flux-conserving'``: a slower algorithm based on that used in `Montage
  <http://montage.ipac.caltech.edu/index.html>`_. This uses intersection of
  spherical polygons to determine how to redistribute the flux. This method is
  only suitable for celestial images. At this point, this mode is experimental
  and does not yet return results that are exactly the same as Montage.

.. note:: the reprojection/resampling is always done assuming that the image is
          in surface brightness units. For example, if one has an iamge with a
          constant value of 1, reprojecting the image to an image with twice as
          high resolution will result in an image where all pixels are all 1.

Reference/API
=============

.. automodapi:: reproject
   :no-inheritance-diagram:

.. automodapi:: reproject.interpolation
   :no-inheritance-diagram:

.. automodapi:: reproject.spherical_intersect
   :no-inheritance-diagram:

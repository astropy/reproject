.. _reprojecting-images:

*********************
Reprojecting an image
*********************

The most common use of *reproject* is to reproject an image from one WCS to
another (for higher-dimensional data, see also :doc:`cubes`). All of the
reprojection algorithms implemented in *reproject* are
available as functions named ``reproject_<algorithm>``, which can be imported
from the top level of the package and called in the same way::

    >>> from reproject import reproject_interp
    >>> from reproject import reproject_adaptive
    >>> from reproject import reproject_exact

.. _common:

Calling the reprojection functions
==================================

The reprojection functions take two main arguments: the data to reproject
(including its WCS information), and the WCS to reproject it to. A number of
input formats are supported, including FITS files, HDU objects, and plain
arrays with a WCS or header - see :ref:`input-formats` and
:ref:`output-projection` for the full details.

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

We can then write out the reprojected image to a new FITS file::

    >>> fits.writeto('reprojected_image.fits', new_image, new_header)   # doctest: +REMOTE_DATA

.. _interpolation:

Choosing the algorithm
======================

The same call as above can be used with any of the reprojection functions -
which one to use depends on the trade-off between speed and accuracy that is
appropriate for your use case, and is discussed in detail in
:ref:`choosing-algorithm`. In short:

* :func:`~reproject.reproject_interp` reprojects using simple interpolation
  and is the fastest option. The order of the interpolation can be controlled
  with the ``order=`` argument (see :ref:`interpolation-order`).

* :func:`~reproject.reproject_adaptive` carries out anti-aliased resampling
  using the `DeForest (2004)
  <https://doi.org/10.1023/B:SOLA.0000021743.24248.b0>`_ algorithm, which
  provides high-quality photometry, in particular when the input and output
  images have different resolutions, and it offers a flux-conserving mode.
  This algorithm has a number of specific options, described in
  :ref:`adaptive-options`.

* :func:`~reproject.reproject_exact` carries out 'exact' reprojection using
  the spherical polygon intersection of input and output pixels. For this
  algorithm, the footprint array returned gives the exact fractional overlap
  of new pixels with the original image (see :ref:`footprints` for more
  details).

.. warning:: The :func:`~reproject.reproject_exact` function is known to
             have precision issues for images with resolutions below
             ~1e-6 arcsec, and will emit a warning in this case, so it is
             best to avoid using it with such images.

Non-celestial data
==================

While reprojecting images of the sky is the most common use case, the
:func:`~reproject.reproject_interp` and :func:`~reproject.reproject_adaptive`
functions work with any WCS - the coordinates do not need to be celestial. For
example, an image with spectral and temporal axes can be reprojected in
exactly the same way as above. The exception is
:func:`~reproject.reproject_exact`, which computes the overlap of pixels as
spherical polygons on the sky and therefore requires a 2-dimensional
celestial WCS.

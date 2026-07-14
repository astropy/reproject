****************************
Common arguments and options
****************************

This page describes the arguments and options that are common to all of the
reprojection functions. For options specific to
:func:`~reproject.reproject_adaptive`, see :ref:`adaptive-options`.

.. _input-formats:

Input data
==========

The first argument to the reprojection functions is the data to reproject,
including WCS information about the data. This can be:

* The name of a FITS file as a `str` or a `pathlib.Path` object
* An :class:`~astropy.io.fits.HDUList` object
* An image HDU object such as a :class:`~astropy.io.fits.PrimaryHDU`,
  :class:`~astropy.io.fits.ImageHDU`, or
  :class:`~astropy.io.fits.CompImageHDU` instance
* A tuple where the first element is a :class:`~numpy.ndarray` and the
  second element is either a :class:`~astropy.wcs.WCS` or a
  :class:`~astropy.io.fits.Header` object
* An :class:`~astropy.nddata.NDData` object from which the ``.data`` and
  ``.wcs`` attributes will be used as the input data
* The name of a PNG or JPEG file with `AVM
  <https://www.virtualastronomy.org/avm_metadata.php>`_ metadata

In the case of a FITS file or an :class:`~astropy.io.fits.HDUList` object, if
there is more than one Header-Data Unit (HDU), the ``hdu_in`` keyword argument
is also required and should be set to the ID or the name of the HDU to use.

The functions in the **reproject.mosaicking** sub-package take a list of
datasets rather than a single one, where each element of the list can be any
of the above.

.. _output-projection:

Output projection
=================

The second argument is the WCS information for the output data, which should
be specified either as a :class:`~astropy.wcs.WCS` or a
:class:`~astropy.io.fits.Header` instance. If this is specified as a
:class:`~astropy.wcs.WCS` instance, the ``shape_out`` keyword argument should
also be specified, and be given the shape of the output data using the Numpy
``(ny, nx)`` convention (this is because :class:`~astropy.wcs.WCS`, unlike
:class:`~astropy.io.fits.Header`, does not contain information about the
data shape).

Return values
=============

The reprojection functions return two arrays - the first is the reprojected
input image, and the second is a 'footprint' array which shows the fraction of
overlap of the input image on the output image grid (see :ref:`footprints`
for more details). To return only the main array and not the footprint, you
can set ``return_footprint=False``.

.. _interpolation-order:

Interpolation order
===================

For :func:`~reproject.reproject_interp`, the order of the interpolation can
be controlled by setting the ``order=`` argument to either an integer or a
string giving the order of the interpolation. Supported strings include:

* ``'nearest-neighbor'``: zeroth order interpolation
* ``'bilinear'``: first order interpolation
* ``'biquadratic'``: second order interpolation
* ``'bicubic'``: third order interpolation

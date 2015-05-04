*******************************
Image reprojection (resampling)
*******************************

Introduction
============

The *reproject* package implements image reprojection (resampling) methods
for astronomical images and more generally n-dimensional data. These assume
that the WCS information contained in the data are correct. This package does
**not** do image registration, which is the process of aligning images where
one or more image may have incorrect or missing WCS.

Requirements
============

This package has the following hard dependencies:
 
* `Numpy <http://www.numpy.org/>`__ 1.6 or later

* `Astropy <http://www.astropy.org/>`__ 1.0 or later

and the following optional dependencies:

* `Scipy <http://www.scipy.org/>`__ for interpolation

* `healpy <http://healpy.readthedocs.org>`_ for HEALPIX image reprojection

Quick start
===========

A common use case is that you have two FITS images, and want to reproject one
to the same header as the other. This can easily be done with the *reproject*
package::

    # Read in data using Astropy
    from astropy.io import fits
    hdu1 = fits.open('image1.fits')
    hdu2 = fits.open('image2.fits')

    # Reproject using the 'reproject' package
    from reproject import reproject_interpolation
    array, footprint = reproject_interpolation(hdu1, hdu2.header)
    
    # Write out reprojected image using Astropy
    fits.writeto('image1_reprojected.fits', array, hdu2.header, clobber=True)

The :func:`~reproject.reproject_interpolation` function above returns the
reprojected array as well as an array that provides information on the
footprint of the first image in the new reprojected image plane.

The *reproject* package supports a number of different algorithms for
reprojection (interpolation, flux-conserving reprojection, etc.) and
different types of data (images, spectral cubes, HEALPIX images, etc.). For
more information, we encourage you to read the full documentation below!

Documentation
=============

The reproject package consists of a few high-level functions to do
reprojection using different algorithms, which depend on the type of data
that you want to reproject.

.. toctree::
   :maxdepth: 2
   
   celestial
   healpix
   noncelestial

Reference/API
=============

.. automodapi:: reproject
   :no-inheritance-diagram:


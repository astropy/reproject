Image reprojection (resampling) (`reproject`)
=============================================

Introduction
------------

The `reproject` package implements image reprojection (resampling) methods
for astronomical images.


.. note::

    We plan to propose that `reproject` will be merged into the
    ``astropy`` core as ``astropy.reproject`` once the main functionality
    is in place and has been tested for a while.

.. note::

    `reproject` requires `numpy <http://www.numpy.org/>`__ and
    `astropy <http://www.astropy.org/>`__ to be installed.
    Some functionality is only available if `scipy <http://www.scipy.org/>`__ or
    `scikit-image <http://scikit-image.org/>`__ are installed, users are
    encouraged to install those optional dependencies.

Getting Started
---------------

We just started, no functionality has been implemented yet, except this:

  >>> import reproject


Using `reproject`
-----------------

.. toctree::
    :maxdepth: 2

    api

.. _SourceExtractor: http://www.astromatic.net/software/sextractor

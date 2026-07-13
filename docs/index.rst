.. _reprojection:

****************************************
Image and cube reprojection (resampling)
****************************************

The *reproject* package implements reprojection (resampling) methods for
astronomical images, spectral cubes, and other n-dimensional data. These
assume that the WCS information contained in the data are correct. This
package does **not** do image registration, which is the process of aligning
images where one or more images may have incorrect or missing WCS.

You can install *reproject* with `pip <http://www.pip-installer.org/en/latest/>`_::

    pip install reproject

or with `conda <https://continuum.io/>`_::

    conda install -c conda-forge reproject

The *reproject* package supports a number of different algorithms for
reprojection (interpolation, flux-conserving reprojection, etc.) and
different types of data (images, spectral cubes, HEALPIX images, etc.). For
more information, we encourage you to read the full documentation below!


Tutorials
---------

.. toctree::
   :maxdepth: 1

   tutorials/first_reprojection
   tutorials/first_mosaicking

How-to guides
-------------

.. toctree::
   :maxdepth: 1

   howto/images
   howto/cubes
   howto/align_north
   howto/dimensions
   howto/multiple_images
   howto/healpix
   howto/hips
   howto/mosaicking
   howto/performance
   howto/chunked
   howto/dask

Explanation
-----------

.. toctree::
   :maxdepth: 1

   explanation/footprints
   explanation/algorithms
   explanation/background_matching

Reference
---------

.. toctree::
   :maxdepth: 1

   reference/options
   reference/adaptive_options
   reference/hips_options
   reference/api

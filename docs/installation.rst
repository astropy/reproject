.. _installation:

********************
Installing reproject
********************

Requirements
============

This package has the following dependencies:

* `Python <http://www.python.org/>`__ 3.8 or later

* `Numpy <http://www.numpy.org/>`__ 1.20 or later

* `Astropy <http://www.astropy.org/>`__ 5.0 or later

* `Scipy <http://www.scipy.org/>`__ 1.5 or later

* `astropy-healpix <https://astropy-healpix.readthedocs.io>`_ 0.6 or later for HEALPIX image reprojection

* `dask <https://www.dask.org/>`_ 2021.8 or later

* `zarr <https://zarr.readthedocs.io/en/stable/>`_

* `fsspec <https://filesystem-spec.readthedocs.io/en/latest/>`_

and the following optional dependencies:

* `shapely <https://toblerity.org/shapely/project.html>`_ 1.6 or later for some of the mosaicking functionality

Installation
============

Using pip
---------

To install *reproject* with `pip <https://pip.pypa.io/en/stable/>`_,
run::

    pip install reproject

Using conda
-----------

To install *reproject* with `conda <https://docs.conda.io/en/latest/>`_, run::

    conda install -c conda-forge reproject

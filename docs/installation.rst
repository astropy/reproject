.. _installation:

********************
Installing reproject
********************

Requirements
============

This package has the following hard run time dependencies:

* `Python <http://www.python.org/>`__ 3.9 or later

* `Numpy <http://www.numpy.org/>`__ 1.20 or later

* `Astropy <http://www.astropy.org/>`__ 5.0 or later

* `Scipy <http://www.scipy.org/>`__ 1.5 or later

* `astropy-healpix <https://astropy-healpix.readthedocs.io>`_ 0.6 or later for HEALPIX image reprojection

and the following optional dependencies:

* `shapely <https://toblerity.org/shapely/project.html>`_ 1.6 or later for some of the mosaicking functionality

and to run the tests, you will also need:

* `Matplotlib <http://matplotlib.org/>`__

* `pytest-arraydiff <https://github.com/astrofrog/pytest-fits>`__

* `pytest-astropy <https://github.com/astropy/pytest-astropy>`__

* `pytest-doctestplus <https://github.com/astropy/pytest-doctestplus>`__


Installation
============

Using pip
---------

To install *reproject* with `pip <http://www.pip-installer.org/en/latest/>`_,
simply run:

    pip install reproject

Using conda
-----------

To install *reproject* with `anaconda <https://continuum.io/>`_, simply run::

    conda install -c conda-forge reproject

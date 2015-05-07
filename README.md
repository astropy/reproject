About
=====

The 'reproject' package is a Python package to reproject astronomical images using various techniques via a uniform interface. By *reprojection*, we mean the re-gridding of images from one world coordinate system to another (for example changing the pixel resolution, orientation, coordinate system). Currently, we have implemented reprojection of celestial images by interpolation (like [SWARP](http://www.astromatic.net/software/swarp)), as well as by finding the exact overlap between pixels on the celestial sphere (like [Montage](http://montage.ipac.caltech.edu/index.html)). It can also reproject to/from HEALPIX projections by relying on the [healpy](https://github.com/healpy/healpy) package. 

For more information, including on how to install the package, see http://reproject.readthedocs.org

![screenshot](docs/images/index-4.png)

Status
======

[![Build Status](https://travis-ci.org/astrofrog/reproject.png?branch=master)](https://travis-ci.org/astrofrog/reproject) [![Coverage Status](https://coveralls.io/repos/astrofrog/reproject/badge.svg?branch=master)](https://coveralls.io/r/astrofrog/reproject?branch=master) [![asv](http://img.shields.io/badge/benchmarked%20by-asv-green.svg?style=flat)](http://astrofrog.github.io/reproject-benchmarks/)

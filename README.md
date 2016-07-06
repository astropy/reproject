[![Build Status](https://travis-ci.org/astrofrog/reproject.svg?branch=master)](https://travis-ci.org/astrofrog/reproject) [![Build status](https://ci.appveyor.com/api/projects/status/0ifg4xonlyrc6eu4/branch/master?svg=true)](https://ci.appveyor.com/project/Astropy/reproject/branch/master) [![Coverage Status](https://coveralls.io/repos/astrofrog/reproject/badge.svg?branch=master)](https://coveralls.io/r/astrofrog/reproject?branch=master) [![asv](http://img.shields.io/badge/benchmarked%20by-asv-green.svg?style=flat)](http://astrofrog.github.io/reproject-benchmarks/) ![Powered by Astropy Badge](http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat)

About
=====

The 'reproject' package is a Python package to reproject astronomical images using various techniques via a uniform interface. By *reprojection*, we mean the re-gridding of images from one world coordinate system to another (for example changing the pixel resolution, orientation, coordinate system). Currently, we have implemented reprojection of celestial images by interpolation (like [SWARP](http://www.astromatic.net/software/swarp)), as well as by finding the exact overlap between pixels on the celestial sphere (like [Montage](http://montage.ipac.caltech.edu/index.html)). It can also reproject to/from HEALPIX projections by relying on the [healpy](https://github.com/healpy/healpy) package. 

For more information, including on how to install the package, see http://reproject.readthedocs.io

![screenshot](docs/images/index-4.png)

Note on license
===============

The code in this package is released under the BSD license. However, the
functions relating to HEALPIX rely on the
[healpy](https://github.com/healpy/healpy) package, which is GPLv2, so if you
use these functions in your code, you are indirectly using healpy and therefore
will need to abide with the GPLv2 license.

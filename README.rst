|Build Status| |Build status| |Coverage Status| |asv| |Powered by Astropy Badge|

About
=====

The 'reproject' package is a Python package to reproject astronomical
images using various techniques via a uniform interface. By
*reprojection*, we mean the re-gridding of images from one world
coordinate system to another (for example changing the pixel resolution,
orientation, coordinate system). Currently, we have implemented
reprojection of celestial images by interpolation (like
`SWARP <http://www.astromatic.net/software/swarp>`__), as well as by
finding the exact overlap between pixels on the celestial sphere (like
`Montage <http://montage.ipac.caltech.edu/index.html>`__). It can also
reproject to/from HEALPIX projections by relying on the
`astropy-healpix <https://github.com/astropy/astropy-healpix>`__
package.

For more information, including on how to install the package, see
https://reproject.readthedocs.io

.. figure:: https://github.com/astrofrog/reproject/raw/master/docs/images/index-4.png
   :alt: screenshot

.. |Build Status| image:: https://travis-ci.org/astrofrog/reproject.svg?branch=master
   :target: https://travis-ci.org/astrofrog/reproject
.. |Build status| image:: https://ci.appveyor.com/api/projects/status/0ifg4xonlyrc6eu4/branch/master?svg=true
   :target: https://ci.appveyor.com/project/Astropy/reproject/branch/master
.. |Coverage Status| image:: https://coveralls.io/repos/astrofrog/reproject/badge.svg?branch=master
   :target: https://coveralls.io/r/astrofrog/reproject?branch=master
.. |asv| image:: http://img.shields.io/badge/benchmarked%20by-asv-green.svg?style=flat
   :target: http://astrofrog.github.io/reproject-benchmarks/
.. |Powered by Astropy Badge| image:: http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat


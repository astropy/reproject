|Build Status| |Coverage Status| |Powered by Astropy Badge|

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

.. |Build Status| image:: https://dev.azure.com/astropy-project/reproject/_apis/build/status/astropy.reproject?branchName=main
   :target: https://dev.azure.com/astropy-project/reproject/_build/latest?definitionId=3&branchName=main
.. |Build status| image:: https://ci.appveyor.com/api/projects/status/0ifg4xonlyrc6eu4/branch/main?svg=true
   :target: https://ci.appveyor.com/project/Astropy/reproject/branch/main
.. |Coverage Status| image:: https://codecov.io/gh/astropy/reproject/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/astropy/reproject
.. |Powered by Astropy Badge| image:: http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat

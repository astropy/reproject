|Docs| |PyPI| |Build Status| |CI Status| |Coverage Status| |Powered by Astropy Badge|

About
=====

The `reproject` package is a Python package to reproject astronomical
images using various techniques via a uniform interface. By
*reprojection*, we mean the re-gridding of images from one world
coordinate system to another (for example changing the pixel resolution,
orientation, coordinate system). Currently, we have implemented
reprojection of celestial images by interpolation (like
`SWARP <http://www.astromatic.net/software/swarp>`__), by the adaptive and
anti-aliased algorithm of `DeForest (2004)
<https://doi.org/10.1023/B:SOLA.0000021743.24248.b0>`_, and by finding the
exact overlap between pixels on the celestial sphere (like `Montage
<http://montage.ipac.caltech.edu/index.html>`__). It can also reproject to/from
HEALPIX projections by relying on the `astropy-healpix
<https://github.com/astropy/astropy-healpix>`__ package.

For more information, including on how to install the package, see
https://reproject.readthedocs.io

.. figure:: https://github.com/astrofrog/reproject/raw/master/docs/images/index-4.png
   :alt: screenshot

.. |Docs| image:: https://readthedocs.org/projects/reproject/badge/?version=latest
   :target: https://reproject.readthedocs.io/en/latest/?badge=latest
.. |PyPI| image:: https://img.shields.io/pypi/v/reproject.svg
   :target: https://pypi.python.org/pypi/reproject
.. |Build Status| image:: https://dev.azure.com/astropy-project/reproject/_apis/build/status/astropy.reproject?branchName=main
   :target: https://dev.azure.com/astropy-project/reproject/_build/latest?definitionId=3&branchName=main
.. |CI Status| image:: https://github.com/astropy/reproject/workflows/CI/badge.svg
   :target: https://github.com/astropy/reproject/actions
.. |Coverage Status| image:: https://codecov.io/gh/astropy/reproject/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/astropy/reproject
.. |Powered by Astropy Badge| image:: http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat
   :target: https://astropy.org

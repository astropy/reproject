
.. _quickstart:

Reprojecting your first image
=============================

A common use case is that you have two FITS images, and want to reproject one
to the same header as the other. This can easily be done with the *reproject*
package, and we demonstrate this in the following example. We start off by
downloading two example images from `http://data.astropy.org <http://data.astropy.org>`_,
namely a 2MASS K-band image and an MSX band E image of the Galactic center:

.. plot::
   :include-source:
   :nofigs:
   :context: reset

    from astropy.io import fits
    from astropy.utils.data import get_pkg_data_filename
    hdu1 = fits.open(get_pkg_data_filename('galactic_center/gc_2mass_k.fits'))[0]
    hdu2 = fits.open(get_pkg_data_filename('galactic_center/gc_msx_e.fits'))[0]

We can examine the two images (this makes use of astropy's `WCSAxes
<https://docs.astropy.org/en/stable/visualization/wcsaxes/>`_ framework
behind the scenes):

.. plot::
   :include-source:
   :context:

    from astropy.wcs import WCS
    import matplotlib.pyplot as plt

    ax1 = plt.subplot(1,2,1, projection=WCS(hdu1.header))
    ax1.imshow(hdu1.data, origin='lower', vmin=-100., vmax=2000.)
    ax1.coords['ra'].set_axislabel('Right Ascension')
    ax1.coords['dec'].set_axislabel('Declination')
    ax1.set_title('2MASS K-band')

    ax2 = plt.subplot(1,2,2, projection=WCS(hdu2.header))
    ax2.imshow(hdu2.data, origin='lower', vmin=-2.e-4, vmax=5.e-4)
    ax2.coords['glon'].set_axislabel('Galactic Longitude')
    ax2.coords['glat'].set_axislabel('Galactic Latitude')
    ax2.coords['glat'].set_axislabel_position('r')
    ax2.coords['glat'].set_ticklabel_position('r')
    ax2.set_title('MSX band E')

We now reproject the MSX image to be in the same projection as the 2MASS image:

.. plot::
   :include-source:
   :nofigs:
   :context:

    from reproject import reproject_interp
    array, footprint = reproject_interp(hdu2, hdu1.header)

The :func:`~reproject.reproject_interp` function above returns the
reprojected array as well as an array that provides information on the
footprint of the first image in the new reprojected image plane (essentially
which pixels in the new image had a corresponding pixel in the old image). We
can now visualize the reprojected data and footprint:

.. plot::
   :include-source:
   :context: close-figs

    from astropy.wcs import WCS
    import matplotlib.pyplot as plt

    ax1 = plt.subplot(1,2,1, projection=WCS(hdu1.header))
    ax1.imshow(array, origin='lower', vmin=-2.e-4, vmax=5.e-4)
    ax1.coords['ra'].set_axislabel('Right Ascension')
    ax1.coords['dec'].set_axislabel('Declination')
    ax1.set_title('Reprojected MSX band E image')

    ax2 = plt.subplot(1,2,2, projection=WCS(hdu1.header))
    ax2.imshow(footprint, origin='lower', vmin=0, vmax=1.5)
    ax2.coords['ra'].set_axislabel('Right Ascension')
    ax2.coords['dec'].set_axislabel('Declination')
    ax2.coords['dec'].set_axislabel_position('r')
    ax2.coords['dec'].set_ticklabel_position('r')
    ax2.set_title('MSX band E image footprint')

We can then write out the image to a new FITS file. Note that, as for
plotting, we can use the header from the 2MASS image since both images are
now in the same projection:

.. plot::
   :include-source:
   :nofigs:
   :context:

   fits.writeto('msx_on_2mass_header.fits', array, hdu1.header, overwrite=True)

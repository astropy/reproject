****************************
Aligning an image with north
****************************

If you need to reproject a celestial image and have it be lined up with the
'north' of a celestial coordinate system, you can use the
:func:`~reproject.mosaicking.find_optimal_celestial_wcs` function to give you
the target WCS:


.. plot::
   :include-source:
   :nofigs:
   :context: reset

    from astropy.io import fits
    from astropy.utils.data import get_pkg_data_filename
    hdu = fits.open(get_pkg_data_filename('galactic_center/gc_bolocam_gps.fits'))[0]

We can now make use of the :func:`~reproject.mosaicking.find_optimal_celestial_wcs`
function to find a WCS with equivalent resolution and position on the sky, but which
is in the ICRS equatorial system, and by default is lined up to the north of that system:

.. plot::
   :include-source:
   :nofigs:
   :context:

    from reproject.mosaicking import find_optimal_celestial_wcs
    wcs_out, shape_out = find_optimal_celestial_wcs(hdu, frame='icrs')

We can then reproject the original image to this new WCS using e.g. :func:`~reproject.reproject_interp`:

.. plot::
   :include-source:
   :nofigs:
   :context:

    from reproject import reproject_interp
    array_north_aligned = reproject_interp(hdu, wcs_out, shape_out=shape_out, return_footprint=False)

and we can then examine the result:

.. plot::
   :include-source:
   :context:

    import matplotlib.pyplot as plt
    from astropy.wcs import WCS

    ax1 = plt.subplot(1,2,1, projection=WCS(hdu.header))
    ax1.imshow(hdu.data, origin='lower', vmin=-0.4, vmax=4)
    ax1.coords['glon'].set_axislabel('Galactic Longitude')
    ax1.coords['glat'].set_axislabel('Galactic Latitude')
    ax1.set_title('Original image')

    ax2 = plt.subplot(1,2,2, projection=wcs_out)
    ax2.imshow(array_north_aligned, origin='lower', vmin=-0.4, vmax=4)
    ax2.coords['ra'].set_axislabel('Right Ascension')
    ax2.coords['dec'].set_axislabel('Declination')
    ax2.set_title('Equatorial-north aligned')

    plt.tight_layout()

You can similarly align images to e.g. galactic north by specifying
``frame='galactic'``.

.. note:: The :func:`~reproject.mosaicking.find_optimal_celestial_wcs`
          function has a number of options to control the output WCS
          and specifying e.g. ``auto_rotate=True`` will no longer
          produce a north-aligned WCS. For some more advanced examples
          of using :func:`~reproject.mosaicking.find_optimal_celestial_wcs`,
          see the :ref:`mosaicking` documentation.

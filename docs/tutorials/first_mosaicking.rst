Making your first mosaic
========================

In this tutorial, we will combine several individual images from the 2MASS
survey into a single mosaic of the M17 region.

We start off by using the `PyVO <https://pyvo.readthedocs.io>`_ package to
retrieve K-band tiles from the 2MASS survey that overlap with the region
around M17:

.. plot::
   :include-source:
   :nofigs:
   :context: reset

    from astropy.io import fits
    from astropy.coordinates import SkyCoord
    from pyvo.dal import imagesearch

    pos = SkyCoord.from_name('M17')
    table = imagesearch('https://irsa.ipac.caltech.edu/cgi-bin/2MASS/IM/nph-im_sia?type=at&ds=asky&',
                        pos, size=0.25).to_table()
    table = table[(table['band'] == 'K') & (table['format'] == 'image/fits')]
    m17_hdus = [fits.open(row['download'])[0] for row in table]

This gives us a list of FITS HDUs, each containing an image and the
associated WCS information.

Before we can combine the images, we need to decide on the WCS and shape of
the final mosaic. Rather than construct these by hand, we can use the
:func:`~reproject.mosaicking.find_optimal_celestial_wcs` function to find a
WCS that covers all of the input images:

.. plot::
   :include-source:
   :nofigs:
   :context:

    from reproject.mosaicking import find_optimal_celestial_wcs

    wcs_out, shape_out = find_optimal_celestial_wcs(m17_hdus)

We can now reproject all the images to this common WCS and combine them into
a mosaic using the :func:`~reproject.mosaicking.reproject_and_coadd`
function. Since the *reproject* package provides several reprojection
algorithms, we need to say which function should be used to reproject the
individual images - here we use :func:`~reproject.reproject_interp`:

.. plot::
   :include-source:
   :nofigs:
   :context:

    from reproject import reproject_interp
    from reproject.mosaicking import reproject_and_coadd

    array, footprint = reproject_and_coadd(m17_hdus,
                                           wcs_out, shape_out=shape_out,
                                           reproject_function=reproject_interp)

The first value returned is the mosaic itself, and the second is a
'footprint' array which shows how many input images contributed to each
output pixel. We can take a look at both:

.. plot::
   :include-source:
   :align: center
   :context:

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 8))
    ax1 = plt.subplot(1, 2, 1)
    im1 = ax1.imshow(array, origin='lower', vmin=600, vmax=800)
    ax1.set_title('Mosaic')
    ax2 = plt.subplot(1, 2, 2)
    im2 = ax2.imshow(footprint, origin='lower')
    ax2.set_title('Footprint')

This is already a usable mosaic! However, each of the input tiles was
observed under slightly different conditions, so each has a slightly
different background level. If we adjust the stretch of the image, we can
see this as vertical striping in the mosaic. To correct for this, we can ask
:func:`~reproject.mosaicking.reproject_and_coadd` to determine and subtract
a constant offset from each image before combining them, using the
``match_background`` option:

.. plot::
   :include-source:
   :nofigs:
   :context:

    array_bgmatch, _ = reproject_and_coadd(m17_hdus,
                                           wcs_out, shape_out=shape_out,
                                           reproject_function=reproject_interp,
                                           match_background=True)

Comparing the two mosaics with an adjusted stretch shows the difference -
the mosaic made without background matching shows vertical striping,
especially on the left:

.. plot::
   :include-source:
   :align: center
   :context: close-figs

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 8))
    ax1 = plt.subplot(1, 2, 1)
    im1 = ax1.imshow(array, origin='lower', vmin=635, vmax=660)
    ax1.set_title('No background matching')
    ax2 = plt.subplot(1, 2, 2)
    im2 = ax2.imshow(array_bgmatch, origin='lower', vmin=635, vmax=660)
    ax2.set_title('Background matching')

And that's it - you have made your first mosaic! To learn more about
customizing the WCS of the mosaic (for example the coordinate system,
rotation, resolution, or projection) and about the other options for
combining images, see :ref:`mosaicking`. To understand what the values in
the footprint array mean, see :ref:`footprints`, and to find out how the
background corrections we used above are determined, see
:ref:`background-matching`.

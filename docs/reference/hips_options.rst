.. _hips-options:

***********************
HiPS generation options
***********************

This page describes in detail some of the options that can be used to control
:func:`~reproject.hips.reproject_to_hips` (see :ref:`hips` for an
introduction).

.. testsetup::

    >>> from astropy.io import fits
    >>> from astropy.utils.data import get_pkg_data_filename
    >>> from reproject import reproject_interp
    >>> from reproject.hips import reproject_to_hips

Setting the maximum order
=========================

The default behavior of :func:`~reproject.hips.reproject_to_hips` is to automatically pick a sensible
maximum order/depth for the HiPS dataset based on the input data resolution, but it is also possible
to set this explicitly:

    >>> hdu = fits.open(get_pkg_data_filename('galactic_center/gc_2mass_k.fits'))[0]
    >>> reproject_to_hips(hdu,
    ...                   output_directory='gc_2mass_k_level',
    ...                   coord_system_out='equatorial',
    ...                   reproject_function=reproject_interp,
    ...                   level=3)

.. FIXME: need to figure out why we need to re-read the file each time to avoid data parsing error

Setting/overriding properties
=============================

A HiPS dataset contains a ``properties`` file which describes the HiPS dataset.
Some of the parameters are set by :func:`~reproject.hips.reproject_to_hips`
by default. Of these, some cannot be overridden (such as tile size and format),
but others can be overridden or set if they were not present in the first place.
A list of all properties can be found in the `HiPS 1.0 <https://www.ivoa.net/documents/HiPS/20170406/PR-HIPS-1.0-20170406.pdf>`__ standard.

You can set/override properties by passing a dictionary to the ``properties``
parameter:

    >>> hdu = fits.open(get_pkg_data_filename('galactic_center/gc_2mass_k.fits'))[0]
    >>> reproject_to_hips(hdu,
    ...                   output_directory='gc_2mass_k_custom_properties',
    ...                   coord_system_out='equatorial',
    ...                   reproject_function=reproject_interp,
    ...                   properties={'obs_title': 'My favorite dataset',
    ...                               'hips_pixel_cut': '400 1000',
    ...                               'creator_did': 'ivo://centre/P/favorite-dataset'})

Progress bar
============

Depending on the size of the input image and the maximum order/depth of the HiPS data
to be generated, the process of reprojection can in some cases be slow due to the
number of tiles to be generated. To track the progress, you can pass a callable
such as a function, to the ``progress_bar`` option. This callable should take an
iterable and yield each of them in time, and can draw/update the progress bar.
One option is to use the `tqdm <https://tqdm.github.io/>`_ package:

    >>> from tqdm import tqdm
    >>> hdu = fits.open(get_pkg_data_filename('galactic_center/gc_2mass_k.fits'))[0]
    >>> reproject_to_hips(hdu,
    ...                   output_directory='gc_2mass_k_custom_progress',
    ...                   coord_system_out='equatorial',
    ...                   reproject_function=reproject_interp,
    ...                   progress_bar=tqdm)  # doctest: +IGNORE_OUTPUT
    100%|█████████████████████████████████████████████| 6/6 [00:00<00:00,  6.13it/s]

Multi-threading
===============

By default, tiles are computed and written out in a single thread, but it is possible
to enable multi-threading, either by setting ``threads=True`` (which automatically
selects the number of threads), or e.g. ``threads=8`` to set the number of threads
explicitly.

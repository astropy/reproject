.. _multiple-images:

******************************************************
Reprojecting multiple images with the same coordinates
******************************************************

If you have multiple images with the exact same coordinate system (e.g. the
red, green, and blue channels of a color image, or a raw image and a
corresponding processed image) and want to reproject all of them to the same
output frame, you can pass them to the reprojection functions as a single
array with an extra leading dimension, and reproject them in one call. This
makes use of the dimension-handling rules described in :doc:`dimensions`:
since the array has one more dimension than the WCS describes, the extra
leading dimension is treated as representing separate images with the same
coordinates, and the coordinate mapping between input and output pixels is
computed only once and reused for each image.

Reprojecting an RGB image
=========================

A color image is a natural example of images sharing coordinates: the three
RGB channels are perfectly aligned. The reprojection functions can take the
filename of a PNG or JPEG image with `AVM
<https://www.virtualastronomy.org/avm_metadata.php>`_ metadata directly, in
which case the image is loaded as an array of shape ``(3, ny, nx)`` along
with the WCS. As an example, we can download a multiwavelength image of
Messier 81 which is rotated by almost 90 degrees from a conventional north-up
orientation:

.. plot::
   :include-source:
   :nofigs:
   :context: reset

    from astropy.utils.data import get_pkg_data_filename
    filename = get_pkg_data_filename('avm/sig07-009.jpg')

We can use :func:`~reproject.mosaicking.find_optimal_celestial_wcs` (see
:doc:`align_north`) to find a WCS that covers the image and is aligned with
north in the ICRS equatorial frame, and then reproject all three color
channels in a single call:

.. plot::
   :include-source:
   :nofigs:
   :context:

    from reproject import reproject_interp
    from reproject.mosaicking import find_optimal_celestial_wcs

    wcs_out, shape_out = find_optimal_celestial_wcs(filename, frame='icrs')
    rgb, footprint = reproject_interp(filename, wcs_out,
                                      shape_out=(3,) + shape_out)

The resulting ``rgb`` array has shape ``(3, 1611, 1250)`` - the leading
dimension still represents the three color channels, which have all been
reprojected using a single coordinate mapping.

To display the result, we convert the values back to 8-bit integers and move
the color axis to the end. For comparison, we also show the original image,
reading its WCS with the `PyAVM <https://astrofrog.github.io/pyavm/>`_
package (which is what the reprojection functions use behind the scenes) and
flipping the rows of the image since JPEG files are stored starting from the
top row:

.. plot::
   :include-source:
   :context:

    import numpy as np
    import matplotlib.pyplot as plt
    from pyavm import AVM

    wcs_in = AVM.from_image(filename).to_wcs()

    original = plt.imread(filename)[::-1]
    reprojected = np.moveaxis(np.nan_to_num(rgb), 0, -1).clip(0, 255).astype(np.uint8)

    plt.figure(figsize=(10, 5))

    ax1 = plt.subplot(1, 2, 1, projection=wcs_in)
    ax1.imshow(original, origin='lower')
    ax1.coords['ra'].set_axislabel('Right Ascension')
    ax1.coords['dec'].set_axislabel('Declination')
    ax1.coords.grid(color='white', alpha=0.5)
    ax1.set_title('Original')

    ax2 = plt.subplot(1, 2, 2, projection=wcs_out)
    ax2.imshow(reprojected, origin='lower')
    ax2.coords['ra'].set_axislabel('Right Ascension')
    ax2.coords['dec'].set_axislabel('Declination')
    ax2.coords['dec'].set_axislabel_position('r')
    ax2.coords['dec'].set_ticklabel_position('r')
    ax2.coords.grid(color='white', alpha=0.5)
    ax2.set_title('Reprojected, north-up')

    plt.tight_layout()

Stacking images yourself
========================

If your images are not already combined into a single array, you can stack
them yourself. For example, given a raw image and a corresponding
background-subtracted image sharing the same coordinate system:

.. doctest-skip::

    >>> from astropy.io import fits
    >>> raw_image, header_in = fits.getdata('raw_image.fits', header=True)
    >>> bg_subtracted_image = fits.getdata('background_subtracted_image.fits')

We can combine the two images into a single array, adding a leading
dimension:

.. doctest-skip::

    >>> import numpy as np
    >>> image_stack = np.stack((raw_image, bg_subtracted_image))
    >>> image_stack.shape
    (2, 1024, 1024)

The header still describes only the two celestial dimensions, so we can
reproject both images in a single call (here ``header_out`` is a header
describing the desired output projection):

.. doctest-skip::

    >>> from reproject import reproject_adaptive
    >>> reprojected, footprint = reproject_adaptive(
    ...         (image_stack, header_in), header_out)
    >>> reprojected.shape
    (2, 1500, 1300)

The first dimension of the result matches the input stack, so we can unpack
the two reprojected images:

.. doctest-skip::

    >>> reprojected_raw = reprojected[0]
    >>> reprojected_bg_subtracted = reprojected[1]

For :func:`~reproject.reproject_interp` and
:func:`~reproject.reproject_adaptive`, this is approximately twice as fast as
reprojecting the two images separately. For :func:`~reproject.reproject_exact`
the savings are much less significant, as producing the coordinate mapping is a
much smaller portion of the total runtime for this algorithm.

While the reproject functions can accept the name of a FITS file as input, from
which the input data and coordinates are loaded automatically, this multi-image
reprojection feature does not support loading multiple images automatically
from multiple HDUs within one FITS file, as it would be difficult to verify
automatically that the HDUs contain the same exact coordinates. If multiple
HDUs do share coordinates and are to be reprojected together, they must be
separately loaded and combined into a single input array (e.g. using
``np.stack`` as in the above example).

.. _multiple-images:

******************************************************
Reprojecting multiple images with the same coordinates
******************************************************

If you have multiple images with the exact same coordinate system (e.g. a raw
image and a corresponding processed image) and want to reproject all of them to
the same output frame, you can stack the images into a single array and
reproject them in one call, which is faster than reprojecting each image
separately. This makes use of the dimension-handling rules described in
:doc:`dimensions`: since the stacked array has one more dimension than the WCS
describes, the extra leading dimension is treated as representing separate
images with the same coordinates, and the coordinate mapping between input and
output pixels is computed only once and reused for each image.

As an example, we start by loading two images that share the exact same
coordinate system:

.. doctest-skip::

    >>> from astropy.io import fits
    >>> raw_image, header_in = fits.getdata('raw_image.fits', header=True)
    >>> bg_subtracted_image = fits.getdata('background_subtracted_image.fits')

We then combine the two images into a single array, adding a leading
dimension:

.. doctest-skip::

    >>> import numpy as np
    >>> image_stack = np.stack((raw_image, bg_subtracted_image))
    >>> image_stack.shape
    (2, 1024, 1024)

The header still describes only the two celestial dimensions, so the extra
leading dimension is treated as representing separate images sharing the same
coordinates, and we can reproject both images in a single call (here
``header_out`` is a header describing the desired output projection):

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

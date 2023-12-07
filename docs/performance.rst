**********************
Optimizing performance
**********************

Disabling coordinate transformation round-tripping
==================================================

For the interpolation and adaptive algorithms, an optional third argument,
``roundtrip_coords`` is accepted. By default, after coordinates are transformed
from the output plane to the input plane, the input-plane coordinates are
transformed back to the output plane to ensure that the transformation is
defined in both directions. This doubles the amount of
coordinate-transformation work to be done. In speed-critical situations, where
it is known that the coordinate transformation is defined everywhere, this
extra work can be disabled by setting ``roundtrip_coords=False``. (Note that
this is not a question of whether each output pixel maps to an existing *pixel*
in the input image and vice-versa, but whether it maps to a valid *coordinate*
in the coordinate system of the input image---regardless of whether that
coordinate falls within the bounds of the input image.)

Disabling returning the footprint
=================================

If you don't need the output footprint after reprojection, you can set
``return_footprint=False`` to return only the reprojected array. This can save
memory and in some cases computing time:

.. doctest-skip::

    >>> array = reproject_interp(..., return_footprint=False)

Using memory-mapped output arrays
=================================

If you are dealing with a large dataset to reproject, you may be want to
write the reprojected array (and optionally the footprint) to an array of your choice, such as for example
a memory-mapped array. For example:

.. doctest-skip::

    >>> header_out = fits.Header.fromtextfile('cube_header_gal.hdr')
    >>> shape = (header_out['NAXIS3'], header_out['NAXIS2'], header_out['NAXIS1'])
    >>> array_out = np.memmap(filename='output.np', mode='w+',
    ...                       shape=shape, dtype='float32')
    >>> hdu = fits.open('cube_file.fits')
    >>> reproject_interp(hdu, header_out, output_array=array_out,
    ...                  return_footprint=False)

After the call to :func:`~reproject.reproject_interp`, ``array_out`` will contain the reprojected values.
If you set up a memory-mapped array for the footprint you could also do:

.. doctest-skip::


    >>> reproject_interp(hdu, header_out, output_array=array_out,
    ...                  output_footprint=footprint_out)

If you are dealing with FITS files, you can skip the numpy memmap step and use `FITS large file creation
<http://docs.astropy.org/en/stable/generated/examples/io/skip_create-large-fits.html>`_:

.. doctest-skip::

    >>> header_out = fits.Header.fromtextfile('cube_header_gal.hdr')
    >>> header_out.tofile('new_cube.fits')
    >>> shape = tuple(header_out['NAXIS{0}'.format(ii)] for ii in range(1, header_out['NAXIS']+1))
    >>> with open('new_cube.fits', 'rb+') as fobj:
    ...     fobj.seek(len(header_out.tostring()) + (np.product(shape) * np.abs(header_out['BITPIX']//8)) - 1)
    ...     fobj.write(b'\0')
    >>> hdu_out = fits.open('new_cube.fits', mode='update')
    >>> rslt = reproject.reproject_interp(hdu, header_out, output_array=hdu_out[0].data,
    ...                                   return_footprint=False)
    >>> hdu_out.flush()

.. _broadcasting:

Multiple images with the same coordinates
=========================================

If you have multiple images with the exact same coordinate system (e.g. a raw
image and a corresponding processed image) and want to reproject both to the
same output frame, it is faster to compute the coordinate mapping between input
and output pixels only once and re-use this mapping for each reprojection. This
is supported by passing multiple input images as an additional dimension in the
input data array. When the input array contains more dimensions than the input
WCS describes, the extra leading dimensions are taken to represent separate
images with the same coordinates, and the reprojection loops over those
dimensions after computing the pixel mapping. For example:

.. doctest-skip::
    >>> raw_image, header_in = fits.getdata('raw_image.fits', header=True)
    >>> bg_subtracted_image = fits.getdata('background_subtracted_image.fits')
    >>> header_out = # Prepare your desired output projection here
    >>> # Combine the two images into one array
    >>> image_stack = np.stack((raw_image, bg_subtracted_image))
    >>> # We provide a header that describes 2 WCS dimensions, but our input
    >>> # array shape is (2, ny, nx)---the 'extra' first dimension represents
    >>> # separate images sharing the same coordinates.
    >>> reprojected, footprint = reproject.reproject_adaptive(
    ...         (image_stack, header_in), header_out)
    >>> # The shape of `reprojected` is (2, ny', nx')
    >>> reprojected_raw, reprojected_bg_subtracted = reprojected[0], reprojected[1]

For :func:`~reproject.reproject_interp` and
:func:`~reproject.reproject_adaptive`, this is approximately twice as fast as
reprojecting the two images separately. For :func:`~reproject.reproject_exact`
the savings are much less significant, as producing the coordinate mapping is a
much smaller portion of the total runtime for this algorithm.

When the output coordinates are provided as a WCS and a ``shape_out`` tuple,
``shape_out`` may describe the output shape of a single image, in which case
the extra leading dimensions are prepended automatically, or it may include the
extra dimensions, in which case the size of the extra dimensions must match
those of the input data exactly.

While the reproject functions can accept the name of a FITS file as input, from
which the input data and coordinates are loaded automatically, this multi-image
reprojection feature does not support loading multiple images automatically
from multiple HDUs within one FITS file, as it would be difficult to verify
automatically that the HDUs contain the same exact coordinates. If multiple
HDUs do share coordinates and are to be reprojected together, they must be
separately loaded and combined into a single input array (e.g. using
``np.stack`` as in the above example).

Chunk by chunk reprojection
===========================

.. testsetup::

    >>> import numpy as np
    >>> import dask.array as da
    >>> input_array = np.random.random((1024, 1024))
    >>> dask_array = da.from_array(input_array, chunks=(128, 128))
    >>> from astropy.wcs import WCS
    >>> wcs_in = WCS(naxis=2)
    >>> wcs_out = WCS(naxis=2)

When calling one of the reprojection functions, you can specify a block size to use for the
reprojection, and this is used to iterate over chunks in the output array in
chunks. For instance, if you pass in a (1024, 1024) array and specify that the
shape of the output should be a (2048, 2048) array (e.g., via ``shape_out``),
then if you set ``block_size=(256, 256)``::

    >>> from reproject import reproject_interp
    >>> input_array.shape
    (1024, 1024)
    >>> array, footprint = reproject_interp((input_array, wcs_in), wcs_out,
    ...                                     shape_out=(2048, 2048), block_size=(256, 256))

the reprojection will be done in 64 separate output chunks. Note however that
this does not break up the input array into chunks since in the general case any
input pixel may contribute to any output pixel.

Multi-process reprojection
==========================

By default, the iteration over the output chunks is done in a single
process/thread, but you may specify ``parallel=True`` to process these in
parallel. If you do this, reproject will use multiple processes (rather than
threads) to parallelize the computation (this is because the core reprojection
algorithms we use are not currently thread-safe). If you specify
``parallel=True``, then ``block_size`` will be automatically set to a sensible
default, but you can also set ``block_size`` manually for more control. Note
that you can also set ``parallel=`` to an integer to indicate the number of
processes to use.

By default, in parallel mode, the entire input array will be written to a
temporary file that is then memory-mapped - this is to avoid loading the whole
input array into memory in each process. If you are specifying a WCS with fewer
dimensions than the data to be reprojected, as described in :ref:`broadcasting`,
you can set the block size to be such that the block size along the dimensions
being reprojected cover the whole image, while the other dimensions can be
smaller. For example, if you are reprojecting a spectral cube with dimensions
(500, 2048, 2048) where 500 is the number of spectral channels and (2048, 2048)
is the celestial plane, then if you are reprojecting just the celestial part of
the WCS you can specify a block size of (N, 2048, 2048) and this will enable a
separate reprojection mode where the input array is not written to disk but
where the reprojection is done in truly independent chunks with size (N, 2048, 2048).

Input dask arrays
=================

The three main reprojection functions can accept dask arrays as inputs, e.g.
assuming you have already constructed a dask array named ``dask_array``::

    >>> dask_array
    dask.array<array, shape=(1024, 1024), dtype=float64, chunksize=(128, 128), chunktype=numpy.ndarray>

you can pass this in as part of the first argument to one of the reprojection
functions::

    >>> array, footprint = reproject_interp((dask_array, wcs_in), wcs_out,
    ...                                     shape_out=(2048, 2048))

In general however, we cannot benefit much from the chunking of the input arrays
because any input pixel might in principle contribute to any output pixel.
Therefore, for now, when a dask array is passed as input, it is computed using
the current default scheduler and converted to a Numpy memory-mapped array. This
is done efficiently in terms of memory and never results in the whole dataset
being loaded into memory at any given time. However, this does require
sufficient space on disk to store the array.

Output dask arrays
==================

By default, the reprojection functions will do the computation immmediately and
return Numpy arrays for the reprojected array and optionally the footprint (this
is regardless of whether dask or Numpy arrays were passed in, or any of the
parallelization options above). However, by setting ``return_type='dask'``, you
can make the functions delay any computation and return dask arrays::

    >>> array, footprint = reproject_interp((input_array, wcs_in), wcs_out,
    ...                                     shape_out=(2048, 2048), block_size=(256, 256),
    ...                                     return_type='dask')
    >>> array
    dask.array<getitem, shape=(2048, 2048), dtype=float64, chunksize=(256, 256), ...>

You can then compute the array or a section of the array yourself whenever you need, or use the
result in further dask expressions.

.. warning:: The reprojection does not currently work reliably when using multiple threads, so
             it is important to make sure you use a dask scheduler that is not multi-threaded.
             At the time of writing, the default dask scheduler is ``threads``, so the scheduler
             needs to be explicitly set to a different one.

Using dask.distributed
======================

The `dask.distributed <https://distributed.dask.org/en/stable/>`_ package makes it
possible to use distributed schedulers for dask. In order to compute
reprojections with dask.distributed, you should make use of the
``return_type='dask'`` option mentioned above so that you can compute the dask
array once the distributed scheduler has been set up. As mentioned in `Output
dask arrays`_, you should make sure that you limit any cluster to have one
thread per process or the results may be unreliable.

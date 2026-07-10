*********************************
Optimizing speed and memory usage
*********************************

This page describes options that can speed up a reprojection or mosaicking
run or reduce its memory usage. See also :doc:`chunked` for carrying out a
reprojection in chunks and in parallel, and :doc:`dask` for using dask arrays
as input or output.

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

The same applies to mosaicking: if you are producing a large mosaic, you can
write the mosaic and footprint from
:func:`~reproject.mosaicking.reproject_and_coadd` to arrays of your choice in
the same way:

.. doctest-skip::

    >>> output_array = np.memmap(filename='array.np', mode='w+',
    ...                          shape=shape_out, dtype='float32')
    >>> output_footprint = np.memmap(filename='footprint.np', mode='w+',
    ...                              shape=shape_out, dtype='float32')
    >>> reproject_and_coadd(...,
    ...                     output_array=output_array,
    ...                     output_footprint=output_footprint)

Using memory-mapped intermediate arrays when mosaicking
=======================================================

During the mosaicking process, each image is reprojected to the minimal subset
of the final header that it covers. In some cases, this can result in arrays
that may not fit in memory. In this case, you can use the
``intermediate_memmap`` option to indicate that all intermediate arrays in the
mosaicking process should use memory-mapped arrays rather than in-memory
arrays:

.. doctest-skip::

    >>> reproject_and_coadd(...,
    ...                     intermediate_memmap=True)

This option can also be set to ``'zarr'``, in which case the intermediate
arrays are stored as zarr arrays on disk instead, which is typically more
efficient - each image is then reprojected in blocks, and the zarr store is
removed as soon as the image has been combined. Note however that this cannot
be used together with ``match_background=True``.

Combined with the above option to specify the output array and footprint for
the final mosaic, it is possible to make sure that no large arrays are ever
loaded into memory. Note however that you will need to make sure you have
sufficient disk space in your temporary directory - as when reprojecting dask
arrays (see :doc:`dask`), the ``TMPDIR`` environment variable can be used to
point at a directory with sufficient space.

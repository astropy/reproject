***************************
Adaptive resampling options
***************************

This page describes in detail some of the options that are specific to
:func:`~reproject.reproject_adaptive`.

Flux units
==========

A rescaling of output pixel values to conserve flux can be enabled with the
``conserve_flux`` flag (flux conservation is stronger with a Gaussian
kernel---see below). Note that here we mean that if this option is set, the
input image is assumed to be in flux per pixel units rather than surface
brightness, rather than the concept of making the result more photometrically
accurate (see :ref:`flux-conservation` for more details).

Kernel
======

The kernel used for interpolation and averaging can be controlled with a set of
options. The ``kernel`` argument can be set to 'hann' or 'gaussian' to set the
function being used. The Gaussian window is the default, as it provides better
anti-aliasing and photometric accuracy (or flux conservation, when the
flux-conserving mode is enabled), though at the cost of blurring the output
image by a few pixels. The ``kernel_width`` argument sets the width of the
Gaussian kernel, in pixels, and is ignored for the Hann window. This width is
measured between the Gaussian's :math:`\pm 1 \sigma` points. The default value
is 1.3 for the Gaussian, chosen to minimize blurring without compromising
accuracy. Lower values may introduce photometric errors or leave input pixels
under-sampled, while larger values may improve anti-aliasing behavior but will
increase blurring of the output image.

Since the Gaussian function has infinite
extent, it must be truncated. This is done by sampling within a region of
finite size. The width in pixels of the sampling region is determined by the
coordinate transform and scaled by the ``sample_region_width`` option, and this
scaling represents a trade-off between accuracy and computation speed. The
default value of 4 represents a reasonable choice, with errors in extreme cases
typically limited to less than one percent, while a value of 5 typically reduces
extreme errors to a fraction of a percent. (The ``sample_region_width`` option
has no effect for the Hann window, as that window does not have infinite
extent.)

Jacobian
========

One can control the calculation of the Jacobian used in this
algorithm with the ``center_jacobian`` flag. The Jacobian matrix represents
how the corresponding input-image coordinate varies as you move between output
pixels (or d(input image coordinate) / d(output image coordinate)), and serves
as a local linearization of the coordinate transformation. When this flag is
``True``, the Jacobian is calculated at pixel grid points by calculating the
transformation at locations offset by half a pixel, and then doing finite
differences on the resulting input-image coordinates. This is more accurate but
carries the cost of tripling the number of coordinate transformed done by this
routine. This is recommended if your coordinate transform varies significantly
and non-smoothly between output pixels. When ``False``, the Jacobian is
calculated using the pixel-grid-point transforms that need to be computed
anyway, which produces Jacobian values at locations between pixel grid points,
and nearby Jacobian values are averaged to produce values at the pixel grid
points. This is more efficient, and the loss of accuracy is extremely small for
transformations that vary smoothly between pixels. The default (``False``) is
to use the faster option.

In some situations (e.g. an all-sky map, with a wrap point in the longitude),
extremely large Jacobian values may be computed which are artifacts of the
coordinate system definition, rather than reflecting the actual nature of the
coordinate transformation. This may result in a band of ``nan`` pixels in the
output image. In these situations, if the actual transformation is
approximately constant in the region of these artifacts, the
``despike_jacobian`` option should be enabled. If enabled, the typical
magnitude (distance from the determinant) of the Jacobian matrix, ``Jmag2 =
sum_j sum_i (J_ij**2)``, is computed for each pixel and compared to the 25th
percentile of that value in the local 3x3 neighborhood (i.e. the third-lowest
value). If it exceeds that percentile value by more than 10 times, the Jacobian
matrix is deemed to be "spiking" and it is replaced by the average of the
non-spiking values in the 3x3 neighborhood.

Boundary mode
=============

When, for any one output pixel, the sampling region in the input image
straddles the boundary of the input image or lies entirely outside the input
image, a range of boundary modes can be applied, and this is set with the
``boundary_mode`` option. Allowed values are:

* ``strict`` --- Output pixels will be ``NaN`` if any of their input samples
  fall outside the input image.
* ``constant`` --- Samples outside the bounds of the input image are
  replaced by a constant value, set with the ``boundary_fill_value`` argument.
  Output values become ``NaN`` if there are no valid input samples.
* ``grid-constant`` --- Samples outside the bounds of the input image are
  replaced by a constant value, set with the ``boundary_fill_value`` argument.
  Output values will be ``boundary_fill_value`` if there are no valid input
  samples.
* ``ignore`` --- Samples outside the input image are simply ignored,
  contributing neither to the output value nor the sum-of-weights
  normalization. If there are no valid input samples, the output value will be
  ``NaN``.
* ``ignore_threshold`` --- Acts as ``ignore``, unless the total weight that
  would have been assigned to the ignored samples exceeds a set fraction of the
  total weight across the entire sampling region, set by the
  ``boundary_ignore_threshold`` argument. In that case, acts as ``strict``.
* ``nearest`` --- Samples outside the input image are replaced by the nearest
  in-bounds input pixel.

The input image can also be marked as being cyclic or periodic in the x and/or
y axes with the ``x_cyclic`` and ``y_cyclic`` flags. If these are set, samples
will wrap around to the opposite side of the image, ignoring the
``boundary_mode`` for that axis.

Non-finite value handling
=========================

This implementation includes several options for handling ``nan`` and ``inf``
values in the input data, set via the ``bad_value_mode`` argument:

* ``strict`` --- Values of ``nan`` or ``inf`` in the input data are propagated
  to every output value which samples them.
* ``ignore`` --- When a sampled input value is ``nan`` or ``inf``, that input
  pixel is ignored (affected neither the accumulated sum of weighted samples
  nor the accumulated sum of weights).
* ``constant`` --- Input values of ``nan`` and ``inf`` are replaced with a
  constant value, set via the ``bad_fill_value`` argument.

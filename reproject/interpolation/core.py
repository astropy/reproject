# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function

import numpy as np
from astropy import wcs

from distutils.version import LooseVersion
NP_LT_17 = LooseVersion(np.__version__) < LooseVersion('1.7')

from ..wcs_utils import convert_world_coordinates
from ..array_utils import iterate_over_celestial_slices, pad_edge_1


def map_coordinates(image, coords, **kwargs):

    # In the built-in scipy map_coordinates, the values are defined at the
    # center of the pixels. This means that map_coordinates does not
    # correctly treat pixels that are in the outer half of the outer pixels.
    # We solve this by extending the array, updating the pixel coordinates,
    # then getting rid of values that were sampled in the range -1 to -0.5
    # and n to n - 0.5.

    from scipy.ndimage import map_coordinates as scipy_map_coordinates

    image = pad_edge_1(image)

    values = scipy_map_coordinates(image, coords + 1, **kwargs)

    reset = np.zeros(coords.shape[1], dtype=bool)

    for i in range(coords.shape[0]):
        reset |= (coords[i] < -0.5)
        reset |= (coords[i] > image.shape[i] - 0.5)

    values[reset] = kwargs.get('cval', 0.)

    return values


def _get_input_pixels_full(wcs_in, wcs_out, shape_out):
    """
    Get the pixel coordinates of the pixels in an array of shape ``shape_out``
    in the input WCS.
    """
    if NP_LT_17:
        raise NotImplementedError("The grid determination requires numpy >=1.7")

    # Generate pixel coordinates of output image
    p_out_ax = []
    for size in shape_out:
        p_out_ax.append(np.arange(size))

    p_out = np.meshgrid(*p_out_ax, indexing='ij')

    # Convert output pixel coordinates to pixel coordinates in original image
    # (using pixel centers).
    args = tuple(p_out[::-1]) + (0,)
    w_out = wcs_out.wcs_pix2world(*args)

    args = tuple(w_out) + (0,)
    p_in = wcs_in.wcs_world2pix(*args)

    # return x,y,z for consistency with _get_input_pixels_celestial
    return p_in


def _get_input_pixels_celestial(wcs_in, wcs_out, shape_out):
    """
    Get the pixel coordinates of the pixels in an array of shape ``shape_out``
    in the input WCS. This function assumes that there are two celestial
    WCS dimensions (which may differ in terms of coordinate systems) and that
    the remaining dimensions match.
    """

    # TODO: for now assuming that coordinates are spherical, not
    # necessarily the case.

    if not wcs_in.has_celestial:
        raise ValueError("Input WCS does not have a celestial component")

    if not wcs_out.has_celestial:
        raise ValueError("Output WCS does not have a celestial component")

    if wcs_in.wcs.naxis != wcs_out.wcs.naxis:
        raise ValueError("Number of dimensions between input and output WCS should match")

    if len(shape_out) != wcs_out.wcs.naxis:
        raise ValueError("Length of shape_out should match number of dimensions in wcs_out")

    # Generate pixel coordinates of output image. This is reversed because
    # numpy and wcs index in opposite directions.
    pixel_out = np.indices(shape_out)[::-1].astype('float')

    # Convert output pixel coordinates to pixel coordinates in original image
    # (using pixel centers).
    world_out = wcs_out.wcs_pix2world(*(tuple(pixel_out) + (0,)))

    # Now we extract the longitude and latitude from the world_out array, and
    # convert these, before converting back to pixel coordinates.
    lon_out, lat_out = world_out[wcs_out.wcs.lng], world_out[wcs_out.wcs.lat]

    lon_in, lat_in = convert_world_coordinates(lon_out, lat_out, wcs_out, wcs_in)

    # We now make an array of *input* world coordinates, taking into account
    # that the order of the axes may be different. However, we can't do this if
    # any items in axis_types is not unique, so we first have to make sure that
    # is the case.
    if np.any(wcs_in.wcs.axis_types != wcs_out.wcs.axis_types):
        if (len(np.unique(wcs_in.wcs.axis_types)) < wcs_in.wcs.naxis or
                len(np.unique(wcs_out.wcs.axis_types)) < wcs_out.wcs.naxis):
            raise ValueError("axis_types contains non-unique elements, and "
                             "input order does not match output order")

    # We start off by creating an empty array of input world coordinates, and
    # populating it index by index
    world_in = np.zeros_like(world_out)
    axis_types_in = list(wcs_out.wcs.axis_types)
    axis_types_out = list(wcs_out.wcs.axis_types)
    for index_in, axis_type in enumerate(axis_types_in):
        index_out = axis_types_out.index(axis_type)
        if index_in == wcs_in.wcs.lng:
            world_in[index_in] = lon_in
        elif index_in == wcs_in.wcs.lat:
            world_in[index_in] = lat_in
        else:
            world_in[index_in] = world_out[index_out]

    pixel_in = wcs_in.wcs_world2pix(*(tuple(world_in) + (0,)))

    return tuple(pixel_in)


def _reproject_celestial(array, wcs_in, wcs_out, shape_out, order=1):
    """
    Reproject data with celestial axes to a new projection using interpolation.
    """

    # For now, assume axes are independent in this routine

    # Check that WCSs are equivalent
    if ((wcs_in.naxis != wcs_out.naxis or
         (list(wcs_in.wcs.axis_types) != list(wcs_out.wcs.axis_types)) or
         (list(wcs_in.wcs.cunit) != list(wcs_out.wcs.cunit)))):
        raise ValueError("The input and output WCS are not equivalent")

    # We create an output array with the required shape, then create an array
    # that is in order of [rest, lat, lon] where rest is the flattened
    # remainder of the array. We then operate on the view, but this will change
    # the original array with the correct shape.

    array_new = np.zeros(shape_out)

    # TODO: Make this more general, we should check all dimensions that aren't lon/lat
    if len(shape_out) >= 3 and (shape_out[0] != array.shape[0]):

        if ((list(wcs_in.sub([wcs.WCSSUB_SPECTRAL]).wcs.ctype) !=
             list(wcs_out.sub([wcs.WCSSUB_SPECTRAL]).wcs.ctype))):
            raise ValueError("The input and output spectral coordinate types "
                             "are not equivalent.")

        # do full 3D interpolation
        xp_in, yp_in, zp_in = _get_input_pixels_celestial(wcs_in, wcs_out,
                                                          shape_out)
        coordinates = np.array([zp_in.ravel(), yp_in.ravel(), xp_in.ravel()])
        bad_data = ~np.isfinite(array)
        array[bad_data] = 0
        array_new = map_coordinates(array, coordinates, order=order,
                                    cval=np.nan,
                                    mode='constant').reshape(shape_out)

    else:

        xp_in = yp_in = None

        subset = None

        # Loop over slices and interpolate
        for slice_in, slice_out in iterate_over_celestial_slices(array,
                                                                 array_new,
                                                                 wcs_in):

            if xp_in is None:

                # Get position of output pixel centers in input image
                xp_in, yp_in = _get_input_pixels_celestial(wcs_in.celestial,
                                                           wcs_out.celestial,
                                                           slice_out.shape)
                coordinates = np.array([yp_in.ravel(), xp_in.ravel()])

                # Now map_coordinates is actually inefficient in that if we
                # pass it a large array, it will be much slower than a small
                # array, even if we only need to reproject part of the image.
                # So here we can instead check what the bounding box of the
                # requested coordinates are. We allow for a 1-pixel padding
                # because map_coordinates needs this
                jmin, imin = np.floor(coordinates.min(axis=1)) - 1
                jmax, imax = np.ceil(coordinates.max(axis=1)) + 1

                ny, nx = array.shape

                # Check first if we are completely outside the image. If this is
                # the case, we should just give up and return an array full of
                # NaN values
                if imin >= nx or imax < 0 or jmin >= ny or jmax < 0:
                    return array_new * np.nan, array_new.astype(float)

                # Now, we check whether there is any point in defining a subset
                if imin > 0 or imax < nx - 1 or jmin > 0 or jmax < ny - 1:
                    subset = (slice(max(jmin, 0), min(jmax, ny - 1)),
                              slice(max(imin, 0), min(imax, nx - 1)))
                    if imin > 0:
                        coordinates[1] -= imin
                    if jmin > 0:
                        coordinates[0] -= jmin


            # If possible, only consider a subset of the array for reprojection.
            # We have already adjusted the coordinates above.
            if subset is not None:
                slice_in = slice_in[subset]

            # Make sure image is floating point. We do this only now because
            # we want to avoid converting the whole input array if possible
            slice_in = np.asarray(slice_in, dtype=float)

            slice_out[:, :] = map_coordinates(slice_in,
                                              coordinates,
                                              order=order, cval=np.nan,
                                              mode='constant'
                                              ).reshape(slice_out.shape)

    return array_new, (~np.isnan(array_new)).astype(float)



def _reproject_full(array, wcs_in, wcs_out, shape_out, order=1):
    """
    Reproject n-dimensional data to a new projection using interpolation.
    """

    # Make sure image is floating point
    array = np.asarray(array, dtype=float)

    # Check that WCSs are equivalent
    if wcs_in.naxis == wcs_out.naxis and np.any(wcs_in.wcs.axis_types != wcs_out.wcs.axis_types):
        raise ValueError("The input and output WCS are not equivalent")


    # We create an output array with the required shape, then create an array
    # that is in order of [rest, lat, lon] where rest is the flattened
    # remainder of the array. We then operate on the view, but this will change
    # the original array with the correct shape.

    array_new = np.zeros(shape_out)

    xp_in, yp_in, zp_in = _get_input_pixels_full(wcs_in, wcs_out, shape_out)

    coordinates = np.array([p.ravel() for p in (zp_in, yp_in, xp_in)])

    array_new = map_coordinates(array,
                                coordinates,
                                order=order, cval=np.nan,
                                mode='constant'
                                ).reshape(shape_out)

    return array_new, (~np.isnan(array_new)).astype(float)

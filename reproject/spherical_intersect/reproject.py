# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

from astropy.io import fits
from astropy.wcs import WCS

from ..wcs_utils import wcs_to_celestial_frame, convert_world_coordinates

from ._overlap import _compute_overlap, _reproject_loop_wrapper, _reproject_par_func
from ._reproject_core import _reproject_slice

__all__ = ['reproject_celestial']

def _par_func(start,end,ny_in,xp_inout,yp_inout,xw_in,yw_in,xw_out,yw_out,nx_out,ny_out,array,shape_out,):
    import numpy as np
    array_new = np.zeros(shape_out)
    weights = np.zeros(shape_out)
    for i in range(start,end):
        for j in range(ny_in):

            # For every input pixel we find the position in the output image in
            # pixel coordinates, then use the full range of overlapping output
            # pixels with the exact overlap function.

            xmin = int(min(xp_inout[j, i], xp_inout[j, i+1], xp_inout[j+1, i+1], xp_inout[j+1, i]))
            xmax = int(max(xp_inout[j, i], xp_inout[j, i+1], xp_inout[j+1, i+1], xp_inout[j+1, i]))
            ymin = int(min(yp_inout[j, i], yp_inout[j, i+1], yp_inout[j+1, i+1], yp_inout[j+1, i]))
            ymax = int(max(yp_inout[j, i], yp_inout[j, i+1], yp_inout[j+1, i+1], yp_inout[j+1, i]))

            ilon = [[xw_in[j, i], xw_in[j, i+1], xw_in[j+1, i+1], xw_in[j+1, i]][::-1]]
            ilat = [[yw_in[j, i], yw_in[j, i+1], yw_in[j+1, i+1], yw_in[j+1, i]][::-1]]
            ilon = np.radians(np.array(ilon))
            ilat = np.radians(np.array(ilat))

            xmin = max(0, xmin)
            xmax = min(nx_out-1, xmax)
            ymin = max(0, ymin)
            ymax = min(ny_out-1, ymax)

            for ii in range(xmin, xmax+1):
                for jj in range(ymin, ymax+1):

                    olon = [[xw_out[jj, ii], xw_out[jj, ii+1], xw_out[jj+1, ii+1], xw_out[jj+1, ii]][::-1]]
                    olat = [[yw_out[jj, ii], yw_out[jj, ii+1], yw_out[jj+1, ii+1], yw_out[jj+1, ii]][::-1]]
                    olon = np.radians(np.array(olon))
                    olat = np.radians(np.array(olat))

                    # Figure out the fraction of the input pixel that makes it
                    # to the output pixel at this position.

                    overlap, _ = _compute_overlap(ilon, ilat, olon, olat)
                    original, _ = _compute_overlap(ilon, ilat, ilon, ilat)
                    array_new[jj, ii] += array[j, i] * overlap / original
                    weights[jj, ii] += overlap / original
    return array_new, weights

def reproject_celestial(array, wcs_in, wcs_out, shape_out, method = "default", nproc = None):
    """
    Reproject celestial slices from an n-d array from one WCS to another using
    flux-conserving spherical polygon intersection.

    Parameters
    ----------
    array : `~numpy.ndarray`
        The array to reproject
    wcs_in : `~astropy.wcs.WCS`
        The input WCS
    wcs_out : `~astropy.wcs.WCS`
        The output WCS
    shape_out : tuple
        The shape of the output array
    method : string
        The underlying algorithmic implementation to use
    nproc : int or None
        The number of processors to use

    Returns
    -------
    array_new : `~numpy.ndarray`
        The reprojected array
    """

    # Convert input array to float values. If this comes from a FITS, it might have
    # float32 as value type and that can break things in cythin.
    array = array.astype(float)

    # TODO: make this work for n-dimensional arrays
    if wcs_in.naxis != 2:
        raise NotImplementedError("Only 2-dimensional arrays can be reprojected at this time")

    # TODO: at the moment, we compute the coordinates of all of the corners,
    # but we might want to do it in steps for large images.

    # Start off by finding the world position of all the corners of the input
    # image in world coordinates

    ny_in, nx_in = array.shape

    x = np.arange(nx_in + 1.) - 0.5
    y = np.arange(ny_in + 1.) - 0.5

    xp_in, yp_in = np.meshgrid(x, y)

    xw_in, yw_in = wcs_in.wcs_pix2world(xp_in, yp_in, 0)

    # Now compute the world positions of all the corners in the output header

    ny_out, nx_out = shape_out

    x = np.arange(nx_out + 1.) - 0.5
    y = np.arange(ny_out + 1.) - 0.5

    xp_out, yp_out = np.meshgrid(x, y)

    xw_out, yw_out = wcs_out.wcs_pix2world(xp_out, yp_out, 0)

    # Convert the input world coordinates to the frame of the output world
    # coordinates.

    xw_in, yw_in = convert_world_coordinates(xw_in, yw_in, wcs_in, wcs_out)

    # Finally, compute the pixel positions in the *output* image of the pixels
    # from the *input* image.

    xp_inout, yp_inout = wcs_out.wcs_world2pix(xw_in, yw_in, 0)

    if method == "default":
        # Create output image

        array_new = np.zeros(shape_out)
        weights = np.zeros(shape_out)

        for i in range(nx_in):
            for j in range(ny_in):

                # For every input pixel we find the position in the output image in
                # pixel coordinates, then use the full range of overlapping output
                # pixels with the exact overlap function.

                xmin = int(min(xp_inout[j, i], xp_inout[j, i+1], xp_inout[j+1, i+1], xp_inout[j+1, i]))
                xmax = int(max(xp_inout[j, i], xp_inout[j, i+1], xp_inout[j+1, i+1], xp_inout[j+1, i]))
                ymin = int(min(yp_inout[j, i], yp_inout[j, i+1], yp_inout[j+1, i+1], yp_inout[j+1, i]))
                ymax = int(max(yp_inout[j, i], yp_inout[j, i+1], yp_inout[j+1, i+1], yp_inout[j+1, i]))

                ilon = [[xw_in[j, i], xw_in[j, i+1], xw_in[j+1, i+1], xw_in[j+1, i]][::-1]]
                ilat = [[yw_in[j, i], yw_in[j, i+1], yw_in[j+1, i+1], yw_in[j+1, i]][::-1]]
                ilon = np.radians(np.array(ilon))
                ilat = np.radians(np.array(ilat))

                xmin = max(0, xmin)
                xmax = min(nx_out-1, xmax)
                ymin = max(0, ymin)
                ymax = min(ny_out-1, ymax)

                for ii in range(xmin, xmax+1):
                    for jj in range(ymin, ymax+1):

                        olon = [[xw_out[jj, ii], xw_out[jj, ii+1], xw_out[jj+1, ii+1], xw_out[jj+1, ii]][::-1]]
                        olat = [[yw_out[jj, ii], yw_out[jj, ii+1], yw_out[jj+1, ii+1], yw_out[jj+1, ii]][::-1]]
                        olon = np.radians(np.array(olon))
                        olat = np.radians(np.array(olat))

                        # Figure out the fraction of the input pixel that makes it
                        # to the output pixel at this position.

                        overlap, _ = _compute_overlap(ilon, ilat, olon, olat)
                        original, _ = _compute_overlap(ilon, ilat, ilon, ilat)
                        array_new[jj, ii] += array[j, i] * overlap / original
                        weights[jj, ii] += overlap / original

        array_new /= weights

        return array_new

    if method == 'multi_py':
        from multiprocessing import Pool, cpu_count
        nproc = cpu_count() if nproc is None else nproc
        pool = Pool(nproc)

        results = []

        for i in range(nproc):
            start = int(nx_in) // nproc * i
            end = int(nx_in) if i == nproc - 1 else int(nx_in) // nproc * (i + 1)
            results.append(pool.apply_async(_par_func,[start,end,ny_in,xp_inout,yp_inout,xw_in,yw_in,xw_out,yw_out,nx_out,ny_out,array,shape_out]))

        pool.close()
        pool.join()

        array_new = sum([_.get()[0] for _ in results])
        weights = sum([_.get()[1] for _ in results])

        return array_new / weights

    if method == 'cython':
        # Create output image

        array_new = np.zeros(shape_out)
        weights = np.zeros(shape_out)

        _reproject_loop_wrapper(nx_in,ny_in,nx_out,ny_out,xp_inout,yp_inout,xw_in,xw_out,
                                yw_in,yw_out,array_new,weights,array)

        array_new /= weights

        return array_new

    if method == "multi_cy":
        from multiprocessing import Pool, cpu_count
        nproc = cpu_count() if nproc is None else nproc
        pool = Pool(nproc)

        results = []

        for i in range(nproc):
            start = int(nx_in) // nproc * i
            end = int(nx_in) if i == nproc - 1 else int(nx_in) // nproc * (i + 1)
            results.append(pool.apply_async(_reproject_par_func,[start,end,ny_in,xp_inout,yp_inout,xw_in,yw_in,xw_out,yw_out,nx_out,ny_out,array,shape_out]))

        pool.close()
        pool.join()

        array_new = sum([_.get()[0] for _ in results])
        weights = sum([_.get()[1] for _ in results])

        return array_new / weights

    if method == "numba":
        from numba import double
        from numba.decorators import jit, autojit
        _par_func_numba = autojit(_par_func)
        array_new, weights = _par_func(0,nx_in,ny_in,xp_inout,yp_inout,xw_in,yw_in,xw_out,yw_out,nx_out,ny_out,array,shape_out)
        array_new /= weights

        return array_new

    if method == "c":
        # startx,endx,starty,endy,nx_out,ny_out,xp_inout,yp_inout,xw_in,yw_in,xw_out,yw_out,array,shape_out
        array_new, weights = _reproject_slice(0,nx_in,0,ny_in,nx_out,ny_out,xp_inout,yp_inout,xw_in,yw_in,xw_out,yw_out,array,shape_out);

        array_new /= weights

        return array_new

    if method == "multi_c":
        from multiprocessing import Pool, cpu_count
        nproc = cpu_count() if nproc is None else nproc
        pool = Pool(nproc)

        results = []

        for i in range(nproc):
            start = int(nx_in) // nproc * i
            end = int(nx_in) if i == nproc - 1 else int(nx_in) // nproc * (i + 1)
            results.append(pool.apply_async(_reproject_slice,[start,end,0,ny_in,nx_out,ny_out,xp_inout,yp_inout,xw_in,yw_in,xw_out,yw_out,array,shape_out]))

        pool.close()
        pool.join()

        array_new = sum([_.get()[0] for _ in results])
        weights = sum([_.get()[1] for _ in results])

        return array_new / weights

    if method == "new_cython":
        from numpy import ascontiguousarray as aca
        from ._overlap import _reproject_slice_cython
        array_new, weights = _reproject_slice_cython(0,nx_in,0,ny_in,nx_out,ny_out,aca(xp_inout),aca(yp_inout),aca(xw_in),aca(yw_in),aca(xw_out),aca(yw_out),aca(array),shape_out);

        array_new /= weights

        return array_new

    if method == "multi_new_cython":
        from numpy import ascontiguousarray as aca
        from ._overlap import _reproject_slice_cython
        from multiprocessing import Pool, cpu_count
        nproc = cpu_count() if nproc is None else nproc
        pool = Pool(nproc)

        results = []

        for i in range(nproc):
            start = int(nx_in) // nproc * i
            end = int(nx_in) if i == nproc - 1 else int(nx_in) // nproc * (i + 1)
            results.append(pool.apply_async(_reproject_slice_cython,[start,end,0,ny_in,nx_out,ny_out,aca(xp_inout),aca(yp_inout),aca(xw_in),aca(yw_in),aca(xw_out),aca(yw_out),aca(array),shape_out]))

        pool.close()
        pool.join()

        array_new = sum([_.get()[0] for _ in results])
        weights = sum([_.get()[1] for _ in results])

        return array_new / weights

    raise ValueError('unrecognized method "{0}"'.format(method,))

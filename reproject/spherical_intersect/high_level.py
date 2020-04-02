# Licensed under a 3-clause BSD style license - see LICENSE.rst

from ..utils import parse_input_data, parse_output_projection
from ..wcs_utils import has_celestial
from .core import _reproject_celestial

__all__ = ['reproject_exact']


def reproject_exact(input_data, output_projection, shape_out=None, hdu_in=0,
                    parallel=False, return_footprint=True):
    """
    Reproject data to a new projection using flux-conserving spherical
    polygon intersection (this is the slowest algorithm).

    Parameters
    ----------
    input_data
        The input data to reproject. This can be:

            * The name of a FITS file
            * An `~astropy.io.fits.HDUList` object
            * An image HDU object such as a `~astropy.io.fits.PrimaryHDU`,
              `~astropy.io.fits.ImageHDU`, or `~astropy.io.fits.CompImageHDU`
              instance
            * A tuple where the first element is a `~numpy.ndarray` and the
              second element is either a `~astropy.wcs.WCS` or a
              `~astropy.io.fits.Header` object
            * An `~astropy.nddata.NDData` object from which the ``.data`` and
              ``.wcs`` attributes will be used as the input data.

    output_projection : `~astropy.wcs.WCS` or `~astropy.io.fits.Header`
        The output projection, which can be either a `~astropy.wcs.WCS`
        or a `~astropy.io.fits.Header` instance.
    shape_out : tuple, optional
        If ``output_projection`` is a `~astropy.wcs.WCS` instance, the
        shape of the output data should be specified separately.
    hdu_in : int or str, optional
        If ``input_data`` is a FITS file or an `~astropy.io.fits.HDUList`
        instance, specifies the HDU to use.
    parallel : bool or int
        Flag for parallel implementation. If ``True``, a parallel implementation
        is chosen, the number of processes selected automatically to be equal to
        the number of logical CPUs detected on the machine. If ``False``, a
        serial implementation is chosen. If the flag is a positive integer ``n``
        greater than one, a parallel implementation using ``n`` processes is chosen.
    return_footprint : bool
        Whether to return the footprint in addition to the output array.

    Returns
    -------
    array_new : `~numpy.ndarray`
        The reprojected array
    footprint : `~numpy.ndarray`
        Footprint of the input array in the output array. Values of 0 indicate
        no coverage or valid values in the input image, while values of 1
        indicate valid values. Intermediate values indicate partial coverage.
    """

    array_in, wcs_in = parse_input_data(input_data, hdu_in=hdu_in)
    wcs_out, shape_out = parse_output_projection(output_projection, shape_out=shape_out)

    if has_celestial(wcs_in) and wcs_in.pixel_n_dim == 2 and wcs_in.world_n_dim == 2:
        return _reproject_celestial(array_in, wcs_in, wcs_out, shape_out=shape_out,
                                    parallel=parallel, return_footprint=return_footprint)
    else:
        raise NotImplementedError("Currently only data with a 2-d celestial "
                                  "WCS can be reprojected using flux-conserving algorithm")

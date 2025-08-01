import os
import tempfile
import uuid
from pathlib import Path

import astropy.nddata
import dask.array as da
import numpy as np
from astropy.io import fits
from astropy.io.fits import CompImageHDU, HDUList, Header, ImageHDU, PrimaryHDU
from astropy.wcs import WCS
from astropy.wcs.wcsapi import BaseHighLevelWCS, BaseLowLevelWCS
from astropy.wcs.wcsapi.high_level_wcs_wrapper import HighLevelWCSWrapper
from PIL import Image
from pyavm import AVM

__all__ = [
    "parse_input_data",
    "parse_input_shape",
    "parse_input_weights",
    "parse_output_projection",
    "as_transparent_rgb",
]


def _dask_to_numpy_memmap(dask_array, tmp_dir):
    """
    Given a dask array, write it out to disk and load it again as a Numpy
    memmap array.
    """

    # Sometimes compute() has to be called twice to return a Numpy array,
    # so we need to check here if this is the case and call the first compute()
    if isinstance(dask_array.ravel()[0].compute(), da.Array):
        dask_array = dask_array.compute()

    # Cast the dask array to regular float for two reasons - first, zarr 3.0.0
    # and later doesn't support big-endian arrays, and also we need to anyway
    # make a native float memory mapped array below.
    dask_array = dask_array.astype("<f8", copy=False)

    # NOTE: here we use a new TemporaryDirectory context manager for the zarr
    # array because we can remove the temporary directory straight away after
    # converting the input to a Numpy memory mapped array.

    with tempfile.TemporaryDirectory() as zarr_tmp:
        # First compute and store the dask array to zarr using whatever
        # the default scheduler is at this point
        try:
            dask_array.to_zarr(zarr_tmp)
        except ValueError:
            dask_array.rechunk().to_zarr(zarr_tmp)

        # Load the array back to dask
        zarr_array = da.from_zarr(zarr_tmp)

        # Then store this in the Numpy memmap array (schedulers other than
        # synchronous don't work well when writing to a Numpy array)

        memmap_path = os.path.join(tmp_dir, f"{uuid.uuid4()}.npy")

        memmapped_array = np.memmap(
            memmap_path,
            dtype=float,
            shape=zarr_array.shape,
            mode="w+",
        )

        da.store(
            zarr_array,
            memmapped_array,
            compute=True,
            scheduler="synchronous",
        )

    return memmap_path, memmapped_array


def hdu_to_numpy_memmap(hdu):
    """
    Given an HDU object, return a regular Numpy memmap rather than a Numpy
    array backed by a memmapped buffer as returned by astropy.
    """

    if (
        hdu.header.get("BSCALE", 1) != 1
        or hdu.header.get("BZERO", 0) != 0
        or hdu.fileinfo() is None
        or hdu._data_replaced
        or hdu.fileinfo()["file"].compression is not None
    ):
        return hdu.data

    return np.memmap(
        hdu.fileinfo()["file"].name,
        mode="r",
        dtype=hdu.data.dtype.newbyteorder(">"),
        shape=hdu.data.shape,
        offset=hdu.fileinfo()["datLoc"],
    )


def parse_input_data(input_data, hdu_in=None, source_hdul=None):
    """
    Parse input data to return a Numpy array and WCS object.
    """

    if isinstance(input_data, str | Path):
        if is_png(input_data) or is_jpeg(input_data):
            data = np.array(Image.open(input_data)).transpose(2, 0, 1)[:, ::-1]
            wcs = AVM.from_image(input_data).to_wcs()
            return data, wcs
        else:
            with fits.open(input_data) as hdul:
                return parse_input_data(hdul, hdu_in=hdu_in)
    elif isinstance(input_data, HDUList):
        if hdu_in is None:
            if len(input_data) > 1:
                raise ValueError(
                    "More than one HDU is present, please specify "
                    "HDU to use with ``hdu_in=`` option"
                )
            else:
                hdu_in = 0
        return parse_input_data(input_data[hdu_in], source_hdul=input_data)
    elif isinstance(input_data, PrimaryHDU | ImageHDU) and not isinstance(input_data, CompImageHDU):
        return (hdu_to_numpy_memmap(input_data), WCS(input_data.header, fobj=source_hdul))
    elif isinstance(input_data, CompImageHDU):
        return (input_data.data, WCS(input_data.header, fobj=source_hdul))
    elif isinstance(input_data, tuple) and isinstance(input_data[0], np.ndarray | da.core.Array):
        if isinstance(input_data[1], Header):
            return input_data[0], WCS(input_data[1])
        else:
            if isinstance(input_data[1], BaseHighLevelWCS):
                return input_data[0], input_data[1]
            else:
                return input_data[0], HighLevelWCSWrapper(input_data[1])
    elif (
        isinstance(input_data, BaseHighLevelWCS)
        and input_data.low_level_wcs.array_shape is not None
    ):
        return input_data.array_shape, input_data
    elif isinstance(input_data, BaseLowLevelWCS) and input_data.array_shape is not None:
        return input_data.array_shape, HighLevelWCSWrapper(input_data)
    elif isinstance(input_data, astropy.nddata.NDDataBase):
        return input_data.data, input_data.wcs
    else:
        raise TypeError(
            "input_data should either be an HDU object or a tuple "
            "of (array, WCS) or (array, Header)"
        )


def parse_input_shape(input_shape, hdu_in=None):
    """
    Parse input shape information to return an array shape tuple and WCS object.
    """

    if isinstance(input_shape, str | Path):
        if is_png(input_shape) or is_jpeg(input_shape):
            shape = np.array(Image.open(input_shape)).transpose(2, 0, 1).shape
            wcs = AVM.from_image(input_shape).to_wcs()
            return shape, wcs
        else:
            with fits.open(input_shape) as hdulist:
                return parse_input_shape(hdulist, hdu_in=hdu_in)
    elif isinstance(input_shape, HDUList):
        if hdu_in is None:
            if len(input_shape) > 1:
                raise ValueError(
                    "More than one HDU is present, please specify "
                    "HDU to use with ``hdu_in=`` option"
                )
            else:
                hdu_in = 0
        return parse_input_shape(input_shape[hdu_in])
    elif isinstance(input_shape, PrimaryHDU | ImageHDU | CompImageHDU):
        return input_shape.shape, WCS(input_shape.header)
    elif isinstance(input_shape, tuple) and isinstance(input_shape[0], np.ndarray | da.core.Array):
        if isinstance(input_shape[1], Header):
            return input_shape[0].shape, WCS(input_shape[1])
        else:
            if isinstance(input_shape[1], BaseHighLevelWCS):
                return input_shape[0].shape, input_shape[1]
            else:
                return input_shape[0].shape, HighLevelWCSWrapper(input_shape[1])
    elif isinstance(input_shape, tuple) and isinstance(input_shape[0], tuple):
        if isinstance(input_shape[1], Header):
            return input_shape[0], WCS(input_shape[1])
        else:
            if isinstance(input_shape[1], BaseHighLevelWCS):
                return input_shape
            else:
                return input_shape[0], HighLevelWCSWrapper(input_shape[1])
    elif (
        isinstance(input_shape, BaseHighLevelWCS)
        and input_shape.low_level_wcs.array_shape is not None
    ):
        return input_shape.low_level_wcs.array_shape, input_shape
    elif isinstance(input_shape, BaseLowLevelWCS) and input_shape.array_shape is not None:
        return input_shape.array_shape, HighLevelWCSWrapper(input_shape)
    elif isinstance(input_shape, astropy.nddata.NDDataBase):
        return input_shape.data.shape, input_shape.wcs
    else:
        raise TypeError(
            "input_shape should either be an HDU object or a tuple "
            "of (array-or-shape, WCS) or (array-or-shape, Header)"
        )


def parse_input_weights(input_weights, hdu_weights=None):
    """
    Parse input weights to return a Numpy array.
    """

    if isinstance(input_weights, str):
        return parse_input_data(fits.open(input_weights), hdu_in=hdu_weights)
    elif isinstance(input_weights, HDUList):
        if hdu_weights is None:
            if len(input_weights) > 1:
                raise ValueError(
                    "More than one HDU is present, please specify "
                    "HDU to use with ``hdu_weights=`` option"
                )
            else:
                hdu_weights = 0
        return parse_input_data(input_weights[hdu_weights])
    elif isinstance(input_weights, PrimaryHDU | ImageHDU | CompImageHDU):
        if "CTYPE1" in input_weights.header:
            # all valid WCSes have CTYPE1 specified, at least
            ww = WCS(input_weights.header)
        else:
            ww = None
        return input_weights.data, ww
    elif isinstance(input_weights, np.ndarray):
        return input_weights, None
    else:
        raise TypeError("input_weights should either be an HDU object or a Numpy array")


def parse_output_projection(output_projection, shape_in=None, shape_out=None, output_array=None):
    if shape_out is None:
        if output_array is not None:
            shape_out = output_array.shape
    elif shape_out is not None and output_array is not None:
        if shape_out != output_array.shape:
            raise ValueError("shape_out does not match shape of output_array")

    if isinstance(output_projection, Header):
        wcs_out = WCS(output_projection)
        try:
            shape_out = [
                output_projection[f"NAXIS{i + 1}"] for i in range(output_projection["NAXIS"])
            ][::-1]
        except KeyError:
            if shape_out is None:
                raise ValueError(
                    "Need to specify shape since output header "
                    "does not contain complete shape information"
                ) from None
    elif isinstance(output_projection, BaseLowLevelWCS | BaseHighLevelWCS):
        if isinstance(output_projection, BaseLowLevelWCS) and not isinstance(
            output_projection, BaseHighLevelWCS
        ):
            wcs_out = HighLevelWCSWrapper(output_projection)
        else:
            wcs_out = output_projection
        if shape_out is None:
            if wcs_out.low_level_wcs.array_shape is not None:
                shape_out = wcs_out.low_level_wcs.array_shape
            else:
                raise ValueError(
                    "Need to specify shape_out when specifying output_projection as WCS object"
                )
    elif isinstance(output_projection, str):
        hdu_list = fits.open(output_projection)
        shape_out = hdu_list[0].data.shape
        header = hdu_list[0].header
        wcs_out = WCS(header)
        hdu_list.close()
    else:
        raise TypeError("output_projection should either be a Header, a WCS object, or a filename")

    if len(shape_out) == 0:
        raise ValueError("The shape of the output image should not be an empty tuple")

    if (
        shape_in is not None
        and len(shape_out) < len(shape_in)
        and len(shape_out) == wcs_out.low_level_wcs.pixel_n_dim
    ):
        # Add the broadcast dimensions to the output shape, which does not
        # currently have any broadcast dims
        shape_out = (*shape_in[: -len(shape_out)], *shape_out)
    return wcs_out, tuple(shape_out)


def is_png(filename):
    with open(filename, "rb") as f:
        return f.read(8) == b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a"


def is_jpeg(filename):
    with open(filename, "rb") as f:
        return f.read(4) == b"\xff\xd8\xff\xe0"


def as_transparent_rgb(data, alpha=None):
    """
    Convert a 3D Numpy array to a PIL object.

    This takes care of swapping the order of the axes, and can apply an additional
    transparency layer. This also converts any NaNs to transparency.

    Parameters
    ----------
    data : `numpy.ndarray`
        Input array with shape ``(3, ny, nx)`` or ``(4, ny, nx)``
    alpha : `numpy.ndarray`
        Alpha layer to apply to the image in addition to any pre-existing alpha
        layer and transparency originating from NaN values. This should have
        a shape ``(ny, nx)``
    """

    if data.ndim != 3:
        raise ValueError("Data needs to be three-dimensional to return RGB image")

    if data.shape[0] not in (3, 4):
        raise ValueError("Data should have shape (3, ny, nx) or (4, ny, nx)")

    array = np.zeros((4,) + data.shape[1:], dtype=np.uint8)

    if alpha is None:

        alpha = np.ones(data.shape[1:])

    else:

        if alpha.ndim != 2:
            raise ValueError("alpha needs to be two-dimensional")

        if alpha.shape != data.shape[1:]:
            raise ValueError(
                "alpha layer shape {alpha.shape} does not match data spatial shape {data.shape[1:]}"
            )

        alpha = alpha.copy()

    alpha[np.any(np.isnan(data), axis=0)] = 0

    data = np.nan_to_num(data)

    if data.shape[0] == 3:
        array[:3] = data
        array[3] = 255
    else:
        array[...] = data

    array[3] = array[3] * alpha

    return Image.fromarray(array.transpose(1, 2, 0)[::-1])

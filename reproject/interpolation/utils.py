import numpy as np

def rebin(data, subsample_factor, wcs=None):
    """
    Upsample a data array by block replication.

    Parameters
    ----------
    data : array_like
        The data to be block replicated.

    subsample_factor : int or tuple (int)
        The subsample factor, if an integer then applied in all N directions, if
        a tuple then length must match number of dimensions of data and will be applied
        appropriately.

    wcs : WCS, optional
        If exists, then will do the sub-sampling with the reproject package.

    Returns
    -------
    output : array_like or tuple of array_like and wcs if the input WCS existed
        The block-replicated data and output WCS if input WCS was included.

    Examples
    --------
    """
    data = np.asanyarray(data)

    if wcs:
        # If the WCS is included as a parameter then we will do the work here.
        import reproject

        w = wcs.deepcopy()
        w.wcs.crpix = wcs.wcs.crpix / np.array(subsample_factor)
        w.wcs.cdelt = wcs.wcs.cdelt * np.array(subsample_factor)

        if isinstance(subsample_factor, (list, tuple)):
            # The subsample factor, if a list/tuple, is in the same order
            # as the w.wcs.ctype, BUT, the data cube is in the opposite
            # order so the subsample factor must be reversed here.
            subsample_factor = np.array(subsample_factor)[::-1]
        else:
            subsample_factor = np.array(subsample_factor)

        output_shape = [int(x) for x in np.array(data.shape) / subsample_factor]

        output_data, output_footprint = reproject.reproject_interp(
            (data, wcs), w, shape_out=output_shape
        )
        return output_data, w
    else:
        # If the WCS is NOT included as a parameter then we will use the
        # map_coordinates function which is similar to what was done
        # in reproject.
        from scipy.ndimage.interpolation import map_coordinates as scipy_map_coordinates

        if isinstance(subsample_factor, (list, tuple)):
            subsample_factor = np.array([float(x) for x in subsample_factor[::-1]])
        else:
            subsample_factor = float(subsample_factor)

        step_size = np.array([int(x) for x in np.array(data.shape) / subsample_factor])

        if len(data.shape) == 2:
            y = np.linspace(0, data.shape[0], step_size[0])
            x = np.linspace(0, data.shape[1], step_size[1])
            xx, yy = np.meshgrid(x, y)
            coords = np.array([yy.flatten(), xx.flatten()])
            output_data = scipy_map_coordinates(data, coords).reshape(xx.shape)
        elif len(data.shape) == 3:
            w = np.linspace(0, data.shape[0], step_size[0])
            x = np.linspace(0, data.shape[1], step_size[1])
            y = np.linspace(0, data.shape[2], step_size[2])
            xx, ww, yy = np.meshgrid(x, w, y)
            coords = np.array([ww.flatten(), xx.flatten(), yy.flatten()])
            output_data = scipy_map_coordinates(data, coords).reshape(xx.shape)
        else:
            raise ValueError('Data size must be be 2D or 3D.')

        return output_data

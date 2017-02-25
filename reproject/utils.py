import numpy as np

from astropy.io import fits
from astropy.io.fits import PrimaryHDU, ImageHDU, CompImageHDU, Header, HDUList
from astropy.wcs import WCS
from astropy.extern import six


def parse_input_data(input_data, hdu_in=None):
    """
    Parse input data to return a Numpy array and WCS object.
    """

    if isinstance(input_data, six.string_types):
        return parse_input_data(fits.open(input_data), hdu_in=hdu_in)
    elif isinstance(input_data, HDUList):
        if len(input_data) > 1 and hdu_in is None:
            raise ValueError("More than one HDU is present, please specify HDU to use with ``hdu_in=`` option")
        return parse_input_data(input_data[hdu_in])
    elif isinstance(input_data, (PrimaryHDU, ImageHDU, CompImageHDU)):
        return input_data.data, WCS(input_data.header)
    elif isinstance(input_data, tuple) and isinstance(input_data[0], np.ndarray):
        if isinstance(input_data[1], Header):
            return input_data[0], WCS(input_data[1])
        else:
            return input_data
    else:
        raise TypeError("input_data should either be an HDU object or a tuple of (array, WCS) or (array, Header)")


def parse_output_projection(output_projection, shape_out=None):

    if isinstance(output_projection, Header):
        wcs_out = WCS(output_projection)
        try:
            shape_out = [output_projection['NAXIS{0}'.format(i + 1)] for i in range(output_projection['NAXIS'])][::-1]
        except KeyError:
            if shape_out is None:
                raise ValueError("Need to specify shape since output header does not contain complete shape information")
    elif isinstance(output_projection, WCS):
        wcs_out = output_projection
        if shape_out is None:
            raise ValueError("Need to specify shape when specifying output_projection as WCS object")

    return wcs_out, shape_out

def flatten(f):
    """Flatten a radio FITS continuum 'cube' so that it becomes a 2D
    image suitable for reprojection. Return new PrimaryHDU"""

    naxis=f[0].header['NAXIS']
    if naxis<2:
        raise Exception('Can\'t make map from this')
    if naxis==2:
        return f[0].header,f[0].data

    w = WCS(f[0].header)
    wn=WCS(naxis=2)
    
    wn.wcs.crpix[0]=w.wcs.crpix[0]
    wn.wcs.crpix[1]=w.wcs.crpix[1]
    wn.wcs.cdelt=w.wcs.cdelt[0:2]
    wn.wcs.crval=w.wcs.crval[0:2]
    wn.wcs.ctype[0]=w.wcs.ctype[0]
    wn.wcs.ctype[1]=w.wcs.ctype[1]
    
    header = wn.to_header()
    header["NAXIS"]=2
    copy=('EQUINOX','EPOCH','BMAJ', 'BMIN', 'BPA', 'RESTFRQ', 'TELESCOP', 'OBSERVER')
    for k in copy:
        r=f[0].header.get(k)
        if r:
            header[k]=r

    slice=[]
    for i in range(naxis,0,-1):
        if i<=2:
            slice.append(np.s_[:],)
        else:
            slice.append(0)
        
    return(fits.PrimaryHDU(header=header,data=f[0].data[slice]))

def find_optimal_header(hdus,imagehdu=None,refhdu=0):
    '''
    Given a list of HDUlists or PrimaryHDUs in hdus, figure out a
    header onto which they can all be reprojected. If they are HDUlists,
    imagehdu must be specified.

    Assumes these are 2D images. To generalize later.

    The reference co-ordinate system is taken from the element refhdu in the list

    '''
    wcs=[]
    data=[]
    for h in hdus:
        d,w = parse_input_data(h,hdu_in=imagehdu)
        wcs.append(w)
        data.append(d)

    cv1 = [w.wcs.crval[0] for w in wcs]
    cv2 = [w.wcs.crval[1] for w in wcs]
    
    mcv1 = np.mean(cv1)
    mcv2 = np.mean(cv2)

    rwcs = wcs[refhdu].deepcopy()
#    rwcs.wcs.ctype = wcs[refhdu].wcs.ctype
#    rwcs.wcs.cdelt = wcs[refhdu].wcs.cdelt
    rwcs.wcs.crval = [mcv1,mcv2]
    rwcs.wcs.crpix = [1,1]
    xmin = xmax = ymin = ymax = 0
    # We now assume that the (non-nan) corners of each image define extreme values
    
    for d,w in zip(data,wcs):
        ys,xs = np.where(d)
        axmin = xs.min()
        aymin = ys.min()
        axmax = xs.max()
        aymax = ys.max()
        del(xs)
        del(ys)

        for x,y in ((axmin,aymin),(axmax,aymin),(axmin,aymax),(axmax,aymax)):
            c1,c2 = [float(f) for f in w.wcs_pix2world(x,y,0)]
            nx,ny = [float(f) for f in rwcs.wcs_world2pix(c1,c2,0)]
            if nx < xmin: xmin = nx
            if nx > xmax: xmax = nx
            if ny < ymin: ymin = ny
            if ny > ymax: ymax = ny

    xsize = int(xmax - xmin)
    ysize = int(ymax - ymin)

    rwcs.wcs.crpix=[1-int(xmin),1-int(ymin)]

    header = rwcs.to_header()
    header['NAXIS'] = 2
    header['NAXIS1'] = xsize
    header['NAXIS2'] = ysize

    return header

def mosaic(hdus,outname,reproject_function,imagehdu=None,refhdu=0,weights=None,copykw=None,use_header=None,overwrite=False,**kwargs):

    '''Combine the list of HDUs in hdus and write the result to outname.
    
    reproject_function is the function to use for the reprojection.
    
    weights, if supplied, can be a list of numerical weights or a list
    of HDUs giving a local weighting factor; if not supplied it's
    taken to be uniform.
    
    refhdu is the reference hdu of the list -- passed to find_optimal_header()

    copykw is a list of keyword names to copy from the reference hdu

    use_header is a header to project on to. If not supplied,
    find_optimal_header() is called.

    overwrite is applied to the writing out outname.

    All other kwargs are passed to the reproject function.

    '''
    if weights is None:
        weights = np.ones(len(hdus))
    if use_header is not None:
        # here's one we prepared earlier
        header = use_header
    else:
        header = find_optimal_header(hdus,imagehdu=imagehdu,refhdu=refhdu)
    xsize,ysize = header['NAXIS1'],header['NAXIS2']
    isum = np.zeros([ysize,xsize])
    wsum = np.zeros_like(isum)
    mask = np.zeros_like(isum,dtype=np.bool)
    for h,w in zip(hdus,weights):
        array, footprint = reproject_function(h,header,**kwargs)
        if isinstance(w,HDUList):
            weight, _ = reproject_function(w,header,**kwargs)
        else:
            weight = w*(~np.isnan(array))
        mask |= ~np.isnan(array)
        weight[np.isnan(array)] = 0
        array[np.isnan(array)] = 0
        isum += array*weight
        wsum += weight

    isum/=wsum
    isum[~mask]=np.nan
    if copykw is not None:
        for kw in copykw:
            try:
                header[kw] = hdus[refhdu].header[kw]
            except KeyError:
                pass
    header['ORIGIN'] = 'reproject-simple-mosaic'
    
    hdu = fits.PrimaryHDU(header=header,data=isum)
    hdu.writeto(outname,overwrite=overwrite)

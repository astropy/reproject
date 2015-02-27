# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import time
from astropy import log
from astropy.io import fits
from . import reproject


def reproject_cmd(infile, targetfile, outfile,
                  inext=0, targetext=0, overwrite=False,
                  method='bilinear', store_footprint=True):
    """
    Reproject one image onto another (file-based interface).

    File-based interface to `reproject.reproject`.

    Parameters
    ----------
    TODO
    """
    log.info('Reading {0}'.format(infile))
    in_hdu_list = fits.open(infile)
    input_data = in_hdu_list[inext]

    log.info('Reading {0}'.format(targetfile))
    target_header = fits.getheader(targetfile, targetext)

    log.info('Reprojecting ...')
    t = time.time()
    result = reproject(input_data=input_data,
                       output_projection=target_header,
                       projection_type=method,
                       )
    t = time.time() - t
    log.info('Done. Wall time: {}'.format(t))


    hdu = fits.PrimaryHDU(data=result[0], header=target_header)
    hdu_list = fits.HDUList(hdu)

    if store_footprint:
        log.info('Coverage map will be stored')
        hdu = fits.ImageHDU(data=result[1], header=target_header, name='coverage')
        hdu_list.append(hdu)
    else:
        log.info('Coverage map will not be stored.')

    log.info('Writing {}'.format(outfile))


def main(args=None):

    from astropy.utils.compat import argparse

    parser = argparse.ArgumentParser(
        description='Reproject one image onto another.')
    # TODO: add link to further info or duplicate it in the description here?
    # http://reproject.readthedocs.org/en/latest/api/reproject.reproject.html
    parser.add_argument('infile',
                        help='Input FITS image filename.')
    parser.add_argument('targetfile',
                        help='Target FITS image filename.')
    parser.add_argument('outfile',
                        help='Output FITS image filename.')
    parser.add_argument('--inext', default=0,
                        help='Input FITS image extension')
    parser.add_argument('--targetext', default=0,
                        help='Target FITS image extension')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing output file?')
    parser.add_argument('--method', type=str, default='bilinear',
                        help='Reprojection method ("nearest-neighbor", "bilinear", '
                        '"biquadratic", "bicubic", or "flux-conserving")')
    parser.add_argument('--store-footprint', default=True, action='store_false',
                        help='Store footprint (=coverage) image in output file?')
    args = parser.parse_args(args)

    reproject_cmd(infile=args.infile,
                  targetfile=args.targetfile,
                  outfile=args.outfile,
                  inext=args.inext,
                  targetext=args.targetext,
                  overwrite=args.overwrite,
                  method=args.method,
                  store_footprint=args.store_footprint,
                  )

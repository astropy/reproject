#ifndef REPROJECT_SLICE_C_H
#define REPROJECT_SLICE_C_H

void _reproject_slice_c(int startx, int endx, int starty, int endy, int nx_out, int ny_out,
    double *xp_inout, double *yp_inout, double *xw_in, double *yw_in, double *xw_out, double *yw_out,
    double *array, double *ilon, double *ilat, double *olon, double * olat, double *array_new, double *weights,
    double *overlap, double *area_ratio, double *original, int col_inout, int col_array, int col_new);

#endif

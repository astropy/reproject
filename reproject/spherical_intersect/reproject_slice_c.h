#ifndef REPROJECT_SLICE_C_H
#define REPROJECT_SLICE_C_H

void _reproject_slice_c(int startx, int endx, int starty, int endy, int nx_out, int ny_out,
    double *xp_inout, double *yp_inout, double *xw_in, double *yw_in, double *xw_out, double *yw_out,
    double *array, double *array_new, double *weights,
    int col_in, int col_out, int col_array, int col_new);

#endif

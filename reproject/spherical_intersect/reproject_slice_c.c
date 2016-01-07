#include "overlapArea.h"
#include "reproject_slice_c.h"

#if defined(_MSC_VER)
  #define INLINE _inline
#else
  #define INLINE inline
#endif

static INLINE double min_4(const double *ptr)
{
    double retval = ptr[0];
    int i;
    for (i = 1; i < 4; ++i) {
        if (ptr[i] < retval) {
            retval = ptr[i];
        }
    }
    return retval;
}

static INLINE double max_4(const double *ptr)
{
    double retval = ptr[0];
    int i;
    for (i = 1; i < 4; ++i) {
        if (ptr[i] > retval) {
            retval = ptr[i];
        }
    }
    return retval;
}

static INLINE double to_rad(double x)
{
    return x * 0.017453292519943295;
}

// Kernel for overlap computation.
static INLINE void _compute_overlap(double *overlap,
                                    double *area_ratio,
                                    double *ilon,
                                    double *ilat,
                                    double *olon,
                                    double *olat)
{
    overlap[0] = computeOverlap(ilon,ilat,olon,olat,0,1,area_ratio);
}

#define GETPTR2(x,ncols,i,j) (x + (i) * (ncols) + (j))
#define GETPTRILON(x,i,j) (x + (j))

void _reproject_slice_c(int startx, int endx, int starty, int endy, int nx_out, int ny_out,
    double *xp_inout, double *yp_inout, double *xw_in, double *yw_in, double *xw_out, double *yw_out,
    double *array, double *array_new, double *weights,
    double *overlap, double *area_ratio, double *original, int col_in, int col_out, int col_array, int col_new)
{
    int i, j, ii, jj, xmin, xmax, ymin, ymax;
    double ilon[4], ilat[4], olon[4], olat[4], minmax_x[4], minmax_y[4];

    // Main loop.
    for (i = startx; i < endx; ++i) {
        for (j = starty; j < endy; ++j) {
            // For every input pixel we find the position in the output image in
            // pixel coordinates, then use the full range of overlapping output
            // pixels with the exact overlap function.

            minmax_x[0] = *GETPTR2(xp_inout,col_in,j,i);
            minmax_x[1] = *GETPTR2(xp_inout,col_in,j,i + 1);
            minmax_x[2] = *GETPTR2(xp_inout,col_in,j + 1,i + 1);
            minmax_x[3] = *GETPTR2(xp_inout,col_in,j + 1,i);

            minmax_y[0] = *GETPTR2(yp_inout,col_in,j,i);
            minmax_y[1] = *GETPTR2(yp_inout,col_in,j,i + 1);
            minmax_y[2] = *GETPTR2(yp_inout,col_in,j + 1,i + 1);
            minmax_y[3] = *GETPTR2(yp_inout,col_in,j + 1,i);

            xmin = (int)(min_4(minmax_x) + .5);
            xmax = (int)(max_4(minmax_x) + .5);
            ymin = (int)(min_4(minmax_y) + .5);
            ymax = (int)(max_4(minmax_y) + .5);

            // Fill in ilon/ilat.
            ilon[0] = to_rad(*GETPTR2(xw_in,col_in,j+1,i));
            ilon[1] = to_rad(*GETPTR2(xw_in,col_in,j+1,i+1));
            ilon[2] = to_rad(*GETPTR2(xw_in,col_in,j,i+1));
            ilon[3] = to_rad(*GETPTR2(xw_in,col_in,j,i));

            ilat[0] = to_rad(*GETPTR2(yw_in,col_in,j+1,i));
            ilat[1] = to_rad(*GETPTR2(yw_in,col_in,j+1,i+1));
            ilat[2] = to_rad(*GETPTR2(yw_in,col_in,j,i+1));
            ilat[3] = to_rad(*GETPTR2(yw_in,col_in,j,i));

            xmin = xmin > 0 ? xmin : 0;
            xmax = (nx_out-1) < xmax ? (nx_out-1) : xmax;
            ymin = ymin > 0 ? ymin : 0;
            ymax = (ny_out-1) < ymax ? (ny_out-1) : ymax;

            for (ii = xmin; ii < xmax + 1; ++ii) {
                for (jj = ymin; jj < ymax + 1; ++jj) {
                    // Fill out olon/olat.
                    olon[0] = to_rad(*GETPTR2(xw_out,col_out,jj+1,ii));
                    olon[1] = to_rad(*GETPTR2(xw_out,col_out,jj+1,ii+1));
                    olon[2] = to_rad(*GETPTR2(xw_out,col_out,jj,ii+1));
                    olon[3] = to_rad(*GETPTR2(xw_out,col_out,jj,ii));

                    olat[0] = to_rad(*GETPTR2(yw_out,col_out,jj+1,ii));
                    olat[1] = to_rad(*GETPTR2(yw_out,col_out,jj+1,ii+1));
                    olat[2] = to_rad(*GETPTR2(yw_out,col_out,jj,ii+1));
                    olat[3] = to_rad(*GETPTR2(yw_out,col_out,jj,ii));

                    // Compute the overlap.
                    _compute_overlap(overlap,area_ratio,ilon,ilat,olon,olat);
                    _compute_overlap(original,area_ratio,olon,olat,olon,olat);

                    // Write into array_new and weights.
                    *GETPTR2(array_new,col_new,jj,ii) += *GETPTR2(array,col_array,j,i) *
                                                                     (overlap[0] / original[0]);

                    *GETPTR2(weights,col_new,jj,ii) += (overlap[0] / original[0]);
                }
            }
        }
    }
}

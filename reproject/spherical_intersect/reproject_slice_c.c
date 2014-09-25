#include "overlapArea.h"
#include "reproject_slice_c.h"

static inline int min_4(const int *ptr)
{
    int retval = ptr[0], i;
    for (i = 1; i < 4; ++i) {
        if (ptr[i] < retval) {
            retval = ptr[i];
        }
    }
    return retval;
}

static inline int max_4(const int *ptr)
{
    int retval = ptr[0], i;
    for (i = 1; i < 4; ++i) {
        if (ptr[i] > retval) {
            retval = ptr[i];
        }
    }
    return retval;
}

static inline double to_rad(double x)
{
    return x * 0.017453292519943295;
}

// Kernel for overlap computation.
static inline void _compute_overlap(double *overlap,
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
    double *array, double *ilon, double *ilat, double *olon, double * olat, double *array_new, double *weights,
    double *overlap, double *area_ratio, double *original, int col_inout, int col_array, int col_new)
{
    int i, j, ii, jj, xmin, xmax, ymin, ymax;

    // Main loop.
    for (i = startx; i < endx; ++i) {
        for (j = starty; j < endy; ++j) {
            // For every input pixel we find the position in the output image in
            // pixel coordinates, then use the full range of overlapping output
            // pixels with the exact overlap function.

            int minmax_x[] = {
                (int)*GETPTR2(xp_inout,col_inout,j,i),
                (int)*GETPTR2(xp_inout,col_inout,j,i + 1),
                (int)*GETPTR2(xp_inout,col_inout,j + 1,i + 1),
                (int)*GETPTR2(xp_inout,col_inout,j + 1,i)
            };

            int minmax_y[] = {
                (int)*GETPTR2(yp_inout,col_inout,j,i),
                (int)*GETPTR2(yp_inout,col_inout,j,i + 1),
                (int)*GETPTR2(yp_inout,col_inout,j + 1,i + 1),
                (int)*GETPTR2(yp_inout,col_inout,j + 1,i)
            };

            xmin = min_4(minmax_x);
            xmax = max_4(minmax_x);
            ymin = min_4(minmax_y);
            ymax = max_4(minmax_y);

            // Fill in ilon/ilat.
            *GETPTRILON(ilon,0,0) = to_rad(*GETPTR2(xw_in,col_inout,j+1,i));
            *GETPTRILON(ilon,0,1) = to_rad(*GETPTR2(xw_in,col_inout,j+1,i+1));
            *GETPTRILON(ilon,0,2) = to_rad(*GETPTR2(xw_in,col_inout,j,i+1));
            *GETPTRILON(ilon,0,3) = to_rad(*GETPTR2(xw_in,col_inout,j,i));

            *GETPTRILON(ilat,0,0) = to_rad(*GETPTR2(yw_in,col_inout,j+1,i));
            *GETPTRILON(ilat,0,1) = to_rad(*GETPTR2(yw_in,col_inout,j+1,i+1));
            *GETPTRILON(ilat,0,2) = to_rad(*GETPTR2(yw_in,col_inout,j,i+1));
            *GETPTRILON(ilat,0,3) = to_rad(*GETPTR2(yw_in,col_inout,j,i));

            xmin = xmin > 0 ? xmin : 0;
            xmax = (nx_out-1) < xmax ? (nx_out-1) : xmax;
            ymin = ymin > 0 ? ymin : 0;
            ymax = (ny_out-1) < ymax ? (ny_out-1) : ymax;

            for (ii = xmin; ii < xmax + 1; ++ii) {
                for (jj = ymin; jj < ymax + 1; ++jj) {
                    // Fill out olon/olat.
                    *GETPTRILON(olon,0,0) = to_rad(*GETPTR2(xw_out,col_inout,jj+1,ii));
                    *GETPTRILON(olon,0,1) = to_rad(*GETPTR2(xw_out,col_inout,jj+1,ii+1));
                    *GETPTRILON(olon,0,2) = to_rad(*GETPTR2(xw_out,col_inout,jj,ii+1));
                    *GETPTRILON(olon,0,3) = to_rad(*GETPTR2(xw_out,col_inout,jj,ii));

                    *GETPTRILON(olat,0,0) = to_rad(*GETPTR2(yw_out,col_inout,jj+1,ii));
                    *GETPTRILON(olat,0,1) = to_rad(*GETPTR2(yw_out,col_inout,jj+1,ii+1));
                    *GETPTRILON(olat,0,2) = to_rad(*GETPTR2(yw_out,col_inout,jj,ii+1));
                    *GETPTRILON(olat,0,3) = to_rad(*GETPTR2(yw_out,col_inout,jj,ii));

                    // Compute the overlap.
                    _compute_overlap(overlap,area_ratio,ilon,ilat,olon,olat);
                    _compute_overlap(original,area_ratio,ilon,ilat,ilon,ilat);

                    // Write into array_new and weights.
                    *GETPTR2(array_new,col_new,jj,ii) += *GETPTR2(array,col_array,j,i) *
                                                                     (overlap[0] / original[0]);

                    *GETPTR2(weights,col_new,jj,ii) += (overlap[0] / original[0]);
                }
            }
        }
    }
}

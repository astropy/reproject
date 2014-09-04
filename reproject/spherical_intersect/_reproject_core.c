#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>

#include "overlapArea.h"

/* Define docstrings */
static char module_docstring[] = "C implementation of utility functions used in reproject";
static char reproject_slice_docstring[] = "Reproject a slice of an image";

/* Declare the C functions here. */
static PyObject *_reproject_slice(PyObject *self, PyObject *args);

/* Define the methods that will be available on the module. */
static PyMethodDef module_methods[] = {
    {"_reproject_slice", _reproject_slice, METH_VARARGS, reproject_slice_docstring},
    {NULL, NULL, 0, NULL}
};

/* This is the function that is called on import. */

#if PY_MAJOR_VERSION >= 3
  #define MOD_ERROR_VAL NULL
  #define MOD_SUCCESS_VAL(val) val
  #define MOD_INIT(name) PyMODINIT_FUNC PyInit_##name(void)
  #define MOD_DEF(ob, name, doc, methods) \
          static struct PyModuleDef moduledef = { \
            PyModuleDef_HEAD_INIT, name, doc, -1, methods, }; \
          ob = PyModule_Create(&moduledef);
#else
  #define MOD_ERROR_VAL
  #define MOD_SUCCESS_VAL(val)
  #define MOD_INIT(name) void init##name(void)
  #define MOD_DEF(ob, name, doc, methods) \
          ob = Py_InitModule3(name, methods, doc);
#endif

MOD_INIT(_reproject_core)
{
    PyObject *m;
    MOD_DEF(m, "_reproject_core", module_docstring, module_methods);
    if (m == NULL)
        return MOD_ERROR_VAL;
    import_array();
    return MOD_SUCCESS_VAL(m);
}

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
static inline void _compute_overlap(PyObject *overlap,
                                    PyObject *area_ratio,
                                    PyObject *ilon,
                                    PyObject *ilat,
                                    PyObject *olon,
                                    PyObject *olat)
{
    int i, n = (int)PyArray_DIMS(ilon)[0];

    for (i = 0; i < n; ++i) {
        *(double *)PyArray_GETPTR1(overlap,i) = computeOverlap((double *)PyArray_GETPTR2(ilon,i,0),
                                                               (double *)PyArray_GETPTR2(ilat,i,0),
                                                               (double *)PyArray_GETPTR2(olon,i,0),
                                                               (double *)PyArray_GETPTR2(olat,i,0),
                                                               0,1,(double *)PyArray_GETPTR1(area_ratio,i));
    }
}

// Function parameters:
// startx,endx,starty,endy,nx_out,ny_out,xp_inout,yp_inout,xw_in,yw_in,xw_out,yw_out,array,shape_out
static PyObject *_reproject_slice(PyObject *self, PyObject *args)
{
    int startx, endx, starty, endy, nx_out, ny_out;
    PyObject *xp_inout_o, *yp_inout_o, *xw_in_o, *yw_in_o,
        *xw_out_o, *yw_out_o, *array_o, *shape_out_o;

    // 6 ints, 8 objs.
    if (!PyArg_ParseTuple(args, "iiiiiiOOOOOOOO", &startx, &endx, &starty, &endy, &nx_out, &ny_out,
        &xp_inout_o, &yp_inout_o, &xw_in_o, &yw_in_o,
        &xw_out_o, &yw_out_o, &array_o, &shape_out_o))
    {
        return NULL;
    }

    // Check the inputs.
    PyObject *xp_inout_a = PyArray_FROM_OTF(xp_inout_o, NPY_DOUBLE, NPY_IN_ARRAY),
        *yp_inout_a = PyArray_FROM_OTF(yp_inout_o, NPY_DOUBLE, NPY_IN_ARRAY),
        *xw_in_a = PyArray_FROM_OTF(xw_in_o, NPY_DOUBLE, NPY_IN_ARRAY),
        *yw_in_a = PyArray_FROM_OTF(yw_in_o, NPY_DOUBLE, NPY_IN_ARRAY),
        *xw_out_a = PyArray_FROM_OTF(xw_out_o, NPY_DOUBLE, NPY_IN_ARRAY),
        *yw_out_a = PyArray_FROM_OTF(yw_out_o, NPY_DOUBLE, NPY_IN_ARRAY),
        *array_a = PyArray_FROM_OTF(array_o, NPY_DOUBLE, NPY_IN_ARRAY);

    if (!xp_inout_a || !yp_inout_a || !xw_in_a || !yw_in_a || !xw_out_a ||
        !yw_out_a || !array_a || !PyTuple_CheckExact(shape_out_o) ||
        PyTuple_Size(shape_out_o) != 2u ||
        !PyInt_CheckExact(PyTuple_GetItem(shape_out_o,0)) ||
        !PyInt_CheckExact(PyTuple_GetItem(shape_out_o,1)) ||
        PyInt_AS_LONG(PyTuple_GetItem(shape_out_o,0)) <= 0 ||
        PyInt_AS_LONG(PyTuple_GetItem(shape_out_o,1)) <= 0)
    {
        PyErr_SetString(PyExc_TypeError, "Invalid input objects.");
        Py_XDECREF(xp_inout_a);
        Py_XDECREF(yp_inout_a);
        Py_XDECREF(xw_in_a);
        Py_XDECREF(yw_in_a);
        Py_XDECREF(xw_out_a);
        Py_XDECREF(yw_out_a);
        Py_XDECREF(array_a);
        return NULL;
    }
    
    // Parse the shape.
    npy_intp shape[2];
    shape[0] = (npy_intp)PyInt_AS_LONG(PyTuple_GetItem(shape_out_o,0));
    shape[1] = (npy_intp)PyInt_AS_LONG(PyTuple_GetItem(shape_out_o,1));

    // Create the array_new and weights objects, plus the objects needed in the loop.
    PyObject *array_new_a = PyArray_SimpleNew(2,shape,NPY_DOUBLE);
    PyObject *weights_a = PyArray_SimpleNew(2,shape,NPY_DOUBLE);

    // ilon/ilat/olon/olat shape.
    npy_intp ll_shape[] = {1,4};
    PyObject *ilon = PyArray_SimpleNew(2,ll_shape,NPY_DOUBLE);
    PyObject *ilat = PyArray_SimpleNew(2,ll_shape,NPY_DOUBLE);
    PyObject *olon = PyArray_SimpleNew(2,ll_shape,NPY_DOUBLE);
    PyObject *olat = PyArray_SimpleNew(2,ll_shape,NPY_DOUBLE);

    // overlap, area_ratio, original.
    npy_intp overlap_shape[] = {PyArray_DIMS(ilon)[0]};
    PyObject *overlap = PyArray_SimpleNew(1,overlap_shape,NPY_DOUBLE);
    PyObject *area_ratio = PyArray_SimpleNew(1,overlap_shape,NPY_DOUBLE);
    PyObject *original = PyArray_SimpleNew(1,overlap_shape,NPY_DOUBLE);
    
    if (!array_new_a || !weights_a || !ilon || !ilat || !olon || !olat || !overlap || !area_ratio || !original) {
        PyErr_SetString(PyExc_MemoryError, "Memory allocation error.");
        Py_XDECREF(xp_inout_a);
        Py_XDECREF(yp_inout_a);
        Py_XDECREF(xw_in_a);
        Py_XDECREF(yw_in_a);
        Py_XDECREF(xw_out_a);
        Py_XDECREF(yw_out_a);
        Py_XDECREF(array_a);
        Py_XDECREF(array_new_a);
        Py_XDECREF(weights_a);
        Py_XDECREF(ilon);
        Py_XDECREF(ilat);
        Py_XDECREF(olon);
        Py_XDECREF(olat);
        Py_XDECREF(overlap);
        Py_XDECREF(area_ratio);
        Py_XDECREF(original);
        return NULL;
    }

    // Fill the arrays with zeroes.
    int i, j, ii, jj;
    for (i = 0; i < shape[0]; ++i) {
        for (j = 0; j < shape[1]; ++j) {
            *(double *)PyArray_GETPTR2(array_new_a,i,j) = 0;
            *(double *)PyArray_GETPTR2(weights_a,i,j) = 0;
        }
    }

    // Main loop.
    int xmin, xmax, ymin, ymax;

    for (i = startx; i < endx; ++i) {
        for (j = starty; j < endy; ++j) {
            // For every input pixel we find the position in the output image in
            // pixel coordinates, then use the full range of overlapping output
            // pixels with the exact overlap function.
            
            int minmax_x[] = {
                (int)*(double *)PyArray_GETPTR2(xp_inout_a,j,i),
                (int)*(double *)PyArray_GETPTR2(xp_inout_a,j,i + 1),
                (int)*(double *)PyArray_GETPTR2(xp_inout_a,j + 1,i + 1),
                (int)*(double *)PyArray_GETPTR2(xp_inout_a,j + 1,i)
            };

            int minmax_y[] = {
                (int)*(double *)PyArray_GETPTR2(yp_inout_a,j,i),
                (int)*(double *)PyArray_GETPTR2(yp_inout_a,j,i + 1),
                (int)*(double *)PyArray_GETPTR2(yp_inout_a,j + 1,i + 1),
                (int)*(double *)PyArray_GETPTR2(yp_inout_a,j + 1,i)
            };

            xmin = min_4(minmax_x);
            xmax = max_4(minmax_x);
            ymin = min_4(minmax_y);
            ymax = max_4(minmax_y);
            
            // Fill in ilon/ilat.
            *(double *)PyArray_GETPTR2(ilon,0,0) = to_rad(*(double *)PyArray_GETPTR2(xw_in_a,j+1,i));
            *(double *)PyArray_GETPTR2(ilon,0,1) = to_rad(*(double *)PyArray_GETPTR2(xw_in_a,j+1,i+1));
            *(double *)PyArray_GETPTR2(ilon,0,2) = to_rad(*(double *)PyArray_GETPTR2(xw_in_a,j,i+1));
            *(double *)PyArray_GETPTR2(ilon,0,3) = to_rad(*(double *)PyArray_GETPTR2(xw_in_a,j,i));

            *(double *)PyArray_GETPTR2(ilat,0,0) = to_rad(*(double *)PyArray_GETPTR2(yw_in_a,j+1,i));
            *(double *)PyArray_GETPTR2(ilat,0,1) = to_rad(*(double *)PyArray_GETPTR2(yw_in_a,j+1,i+1));
            *(double *)PyArray_GETPTR2(ilat,0,2) = to_rad(*(double *)PyArray_GETPTR2(yw_in_a,j,i+1));
            *(double *)PyArray_GETPTR2(ilat,0,3) = to_rad(*(double *)PyArray_GETPTR2(yw_in_a,j,i));

            xmin = xmin > 0 ? xmin : 0;
            xmax = (nx_out-1) < xmax ? (nx_out-1) : xmax;
            ymin = ymin > 0 ? ymin : 0;
            ymax = (ny_out-1) < ymax ? (ny_out-1) : ymax;

            for (ii = xmin; ii < xmax + 1; ++ii) {
                for (jj = ymin; jj < ymax + 1; ++jj) {
                    // Fill out olon/olat.
                    *(double *)PyArray_GETPTR2(olon,0,0) = to_rad(*(double *)PyArray_GETPTR2(xw_out_a,jj+1,ii));
                    *(double *)PyArray_GETPTR2(olon,0,1) = to_rad(*(double *)PyArray_GETPTR2(xw_out_a,jj+1,ii+1));
                    *(double *)PyArray_GETPTR2(olon,0,2) = to_rad(*(double *)PyArray_GETPTR2(xw_out_a,jj,ii+1));
                    *(double *)PyArray_GETPTR2(olon,0,3) = to_rad(*(double *)PyArray_GETPTR2(xw_out_a,jj,ii));

                    *(double *)PyArray_GETPTR2(olat,0,0) = to_rad(*(double *)PyArray_GETPTR2(yw_out_a,jj+1,ii));
                    *(double *)PyArray_GETPTR2(olat,0,1) = to_rad(*(double *)PyArray_GETPTR2(yw_out_a,jj+1,ii+1));
                    *(double *)PyArray_GETPTR2(olat,0,2) = to_rad(*(double *)PyArray_GETPTR2(yw_out_a,jj,ii+1));
                    *(double *)PyArray_GETPTR2(olat,0,3) = to_rad(*(double *)PyArray_GETPTR2(yw_out_a,jj,ii));

                    // Compute the overlap.
                    _compute_overlap(overlap,area_ratio,ilon,ilat,olon,olat);
                    _compute_overlap(original,area_ratio,ilon,ilat,ilon,ilat);

                    // Write into array_new and weights.
                    *(double *)PyArray_GETPTR2(array_new_a,jj,ii) += *(double *)PyArray_GETPTR2(array_a,j,i) *
                                                                     (*(double *)PyArray_GETPTR1(overlap,0) / *(double *)PyArray_GETPTR1(original,0));
                    *(double *)PyArray_GETPTR2(weights_a,jj,ii) += (*(double *)PyArray_GETPTR1(overlap,0) / *(double *)PyArray_GETPTR1(original,0));
                }
            }            
        }
    }

    // Prepare return value.
    PyObject *retval;
    retval = PyTuple_Pack(2,array_new_a,weights_a);

    // Final cleanup.
    Py_XDECREF(xp_inout_a);
    Py_XDECREF(yp_inout_a);
    Py_XDECREF(xw_in_a);
    Py_XDECREF(yw_in_a);
    Py_XDECREF(xw_out_a);
    Py_XDECREF(yw_out_a);
    Py_XDECREF(array_a);
    // NOTE: still need to clean up the return values, as PyTuple_Pack
    // will increase their refcount.
    Py_XDECREF(array_new_a);
    Py_XDECREF(weights_a);
    Py_XDECREF(ilon);
    Py_XDECREF(ilat);
    Py_XDECREF(olon);
    Py_XDECREF(olat);
    Py_XDECREF(overlap);
    Py_XDECREF(area_ratio);
    Py_XDECREF(original);

    // PyTuple_Pack() could return NULL in case of errors (memory allocation maybe?).
    if (!retval) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to build the return tuple from the C function '_reproject_slice()'");
        return NULL;
    }

    return retval;
}

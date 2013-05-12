#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>

/* Define docstrings */
static char module_docstring[] = "Wrap Montage overlap routines";
static char computeOverlap_docstring[] = "Compute spherical polygon overlap";

/* Declare the C functions here. */
static PyObject *_computeOverlap(PyObject *self, PyObject *args);

/* Define the methods that will be available on the module. */
static PyMethodDef module_methods[] = {
    {"_computeOverlap", _computeOverlap, METH_VARARGS, computeOverlap_docstring},
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

MOD_INIT(_overlap_wrapper)
{
    PyObject *m;
    MOD_DEF(m, "_overlap_wrapper", module_docstring, module_methods);
    if (m == NULL)
        return MOD_ERROR_VAL;
    import_array();
    return MOD_SUCCESS_VAL(m);
}

/* Define interfaces */

double computeOverlap(double *, double *, double *, double *, int, double, double *);

/* Do the heavy lifting here */

static PyObject *_computeOverlap(PyObject *self, PyObject *args)
{

    PyObject *ilon_obj, *ilat_obj, *olon_obj, *olat_obj;
    double refArea, areaRatio;
    int energyMode;

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "OOOOidd", &ilon_obj, &ilat_obj, &olon_obj, &olat_obj, &energyMode, &refArea, &areaRatio))
        return NULL;

    /* Interpret the input objects as `numpy` arrays. */
    PyObject *ilon_array = PyArray_FROM_OTF(ilon_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *ilat_array = PyArray_FROM_OTF(ilat_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *olon_array = PyArray_FROM_OTF(olon_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *olat_array = PyArray_FROM_OTF(olat_obj, NPY_DOUBLE, NPY_IN_ARRAY);

    /* If that didn't work, throw an `Exception`. */
    if (ilon_array == NULL || ilat_array == NULL || olon_array == NULL || olat_array == NULL) {
        PyErr_SetString(PyExc_TypeError, "Couldn't parse the input arrays.");
        Py_XDECREF(ilon_array);
        Py_XDECREF(ilat_array);
        Py_XDECREF(olon_array);
        Py_XDECREF(olat_array);
        return NULL;
    }


    /* Get pointers to the data as C-types. */
    double *ilon = (double*)PyArray_DATA(ilon_array);
    double *ilat = (double*)PyArray_DATA(ilat_array);
    double *olon = (double*)PyArray_DATA(olon_array);
    double *olat = (double*)PyArray_DATA(olat_array);

    /* Compute overlap using Montage routine */
    double overlap = computeOverlap(ilon, ilat, olon, olat, energyMode, refArea, &areaRatio);

    /* Build the output tuple */
    PyObject *ret = Py_BuildValue("d", overlap);
    if (ret == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Couldn't build output.");
        return NULL;
    }

    return ret;
}
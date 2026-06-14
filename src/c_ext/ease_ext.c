// pinning the numpy api version prevents deprecation warnings and ensures abi stability across different
// python environments
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include "ease_custom_math.h"
#include "ease_gsl_math.h"
#include <omp.h>

// thin wrapper design: python is strictly responsible for memory contiguity and type enforcement before 
// calling. This this allows the C layer to bypass extensive safety checks and casting overhead during 
// runtime
static PyObject* py_compute_and_invert_gram(PyObject* self, PyObject* args) {
    PyArrayObject *data_arr, *indices_arr, *indptr_arr;
    int n_users, n_items;
    double reg_weight;
    int num_threads;

    // using "O!" format strictly enforces that incoming objects are compiled numpy arrays, instantly 
    // rejecting generic python lists
    if (!PyArg_ParseTuple(args, "O!O!O!iidi", 
                          &PyArray_Type, &data_arr, 
                          &PyArray_Type, &indices_arr, 
                          &PyArray_Type, &indptr_arr, 
                          &n_users, &n_items, &reg_weight,
                          &num_threads)) {
        return NULL;
    }
    // explicitly setting the thread count for the OpenMP
    omp_set_num_threads(num_threads);

    // direct pointer extraction achieves true zero-copy data transfer, it relies on the python layer 
    // to guarantee float64 and int32 c-contiguous memory layouts beforehand
    double* data = (double*)PyArray_DATA(data_arr);
    int* indices = (int*)PyArray_DATA(indices_arr);
    int* indptr = (int*)PyArray_DATA(indptr_arr);

    // allocating the output matrix directly via numpy's C-api, this ensures python's garbage collector 
    // owns the memory and will safely free it when the object is no longer needed
    npy_intp dims[2] = {n_items, n_items};
    PyArrayObject *out_G_arr = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_FLOAT64);
    double* out_G = (double*)PyArray_DATA(out_G_arr);

    // computes X^T X + reg using parallelized C
    compute_gram_matrix_csr(data, indices, indptr, n_users, n_items, reg_weight, out_G);

    // runs gsl cholesky inversion in-place directly on the numpy memory buffer
    invert_gram_matrix_gsl(out_G, n_items);

    return PyArray_Return(out_G_arr);
}

// inference wrapper optimized for high-throughput prediction requests
static PyObject* py_predict_top_k(PyObject* self, PyObject* args) {
    PyArrayObject *B_arr, *interacted_arr;
    int n_items, K;

    // using "O!" format strictly enforces that incoming objects are compiled numpy arrays, instantly 
    // rejecting generic python lists
    if (!PyArg_ParseTuple(args, "O!O!ii", 
                          &PyArray_Type, &B_arr, 
                          &PyArray_Type, &interacted_arr, 
                          &n_items, &K)) {
        return NULL;
    }

    // expecting float32 to minimize memory bandwidth and maximize L1 cache usage during inference
    // scanning
    float* B = (float*)PyArray_DATA(B_arr);
    int* interacted = (int*)PyArray_DATA(interacted_arr);
    
    // extracting the array length dynamically from the numpy metadata struct avoids needing an extra 
    // parameter
    int n_interacted = (int)PyArray_DIM(interacted_arr, 0);

    npy_intp dims[1] = {K};
    PyArrayObject *out_top_k_arr = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_INT32);
    int* out_top_k_indices = (int*)PyArray_DATA(out_top_k_arr);

    // predicts Top-K recommendations
    predict_top_k(B, n_items, interacted, n_interacted, K, out_top_k_indices);

    return PyArray_Return(out_top_k_arr);
}

// mapping the internal C functions to the python method names
static PyMethodDef EaseMethods[] = {
    {"compute_and_invert_gram", py_compute_and_invert_gram, METH_VARARGS, "X^T X + GSL Cholesky Inverse"},
    {"predict_top_k", py_predict_top_k, METH_VARARGS, "Top-K fast inference"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef easemodule = {
    PyModuleDef_HEAD_INIT,
    "ease_core",
    NULL, -1, EaseMethods
};

PyMODINIT_FUNC PyInit_ease_core(void) {
    // crucial requirement for the numpy c-api, omitting this causes a segmentation fault on the first array creation
    import_array(); 
    return PyModule_Create(&easemodule);
}
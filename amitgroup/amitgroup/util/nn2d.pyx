
# cython: boundscheck=False
# cython: wraparound=False
# cython: embedsignature=True
# cython: cdivision=True
# cython: profile=False
import cython
cimport cython
import numpy as np
cimport numpy as np
from math import fabs
#DTYPE = np.float64
#ctypedef np.float64_t DTYPE_t

# cython.numeric does not include uint8
# Can be expended to complex types if needed
ctypedef fused TYPE:
    np.float32_t
    np.float64_t
    np.int16_t
    np.int32_t
    np.int64_t
    np.uint8_t 

def nn_resample2d(np.ndarray[TYPE, ndim=2] X, size):
    """
    Nearest-neighbor resampling of a numpy array. Not sure if there is a built-in alternative for this in numpy/scipy.
    """
    assert len(size) == 2, "size must be iterable of size 2"
    #np.ndarray[cython.numeric, ndim=2] Y = np.empty(size, dtype=X.dtype)
    Y = np.empty(size, dtype=X.dtype)
    cdef:
        TYPE[:,:] X_mv = X
        TYPE[:,:] Y_mv = Y
        
        int dim0 = X.shape[0]
        int dim1 = X.shape[1]
        int new_dim0 = size[0]
        int new_dim1 = size[1]
        double ratio0 = <double>dim0/new_dim0
        double ratio1 = <double>dim1/new_dim1
        int i, j

    with nogil:
        for i in range(new_dim0):
            for j in range(new_dim1):
                Y_mv[i,j] = X_mv[<int>(i*ratio0), <int>(j*ratio1)]

    return Y

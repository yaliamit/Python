# cython compute_likelihood_linear_filter.pyx
# gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing \
#      -I/usr/include/python2.7 -o compute_likelihood_linear_filter.so compute_likelihood_linear_filter.c
#!python
# cython: boundscheck=False
# cython: wraparound=False
# cython: embedsignature=True
import cython
import numpy as np
cimport numpy as np
#from cython.parallel import prange
DTYPE = np.float32
UINT = np.uint8
ctypedef np.float32_t DTYPE_t

ctypedef np.uint8_t UINT_t

def detect(np.ndarray[ndim=3,
                      dtype=UINT_t] F,
           np.ndarray[ndim=3,
                      dtype=DTYPE_t] LF):
    """
    Parameters:
    ===========
    F: np.ndarray[ndim=3,dtype=UINT_t]
        Features for an example utterance that we have fit with the
        intermediate features plus downsampled. Dimensions
        of F and LF are assumed to be (time,frequency,patch_type)
    LF: np.ndarray[ndim=3,dtype=DTYPE_t]
        Linear filter for computing the likelihood
    Output:
    =======
    detect_scores: np.ndarray[ndim=1,dtype=DTYPE_t]
        Performs spreading 
    """
    cdef np.uint16_t F_dim_0 = F.shape[0]
    cdef np.uint16_t F_dim_1 = F.shape[1]
    cdef np.uint16_t F_dim_2 = F.shape[2]
    cdef np.uint16_t LF_dim_0 = LF.shape[0]
    cdef np.uint16_t num_detections = F_dim_0 - LF_dim_0 +1
    cdef np.ndarray[ndim=1,dtype=DTYPE_t] detect_scores = np.zeros(num_detections,
                                                                   dtype=DTYPE)
    cdef np.uint32_t cur_detection,filter_timepoint,frequency,part_identity
    # cur_detection is the time point in the overall vector
    for cur_detection in range(num_detections):
        # filter_timepoint is where we are in the filter for
        # computing these parallel convolutions
        for filter_timepoint in range(LF_dim_0):
            for frequency in range(F_dim_1):
                for part_identity in range(F_dim_2):
                    if F[cur_detection+filter_timepoint,
                         frequency,
                         part_identity]:
                        detect_scores[cur_detection] += (
                        
                            LF[filter_timepoint,
                               frequency,
                               part_identity])
    return detect_scores

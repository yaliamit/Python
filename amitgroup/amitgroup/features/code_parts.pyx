#!python
# cython: boundscheck=False
# cython: wraparound=False
# cython: embedsignature=True
# cython: cdivision=True
import cython
import numpy as np
cimport numpy as np
#from cython.parallel import prange
DTYPE = np.float32
UINT = np.uint8
UINT32 = np.uint32
ctypedef np.float32_t DTYPE_t
ctypedef np.uint8_t UINT_t
ctypedef np.uint32_t UINT32_t

from libc.stdlib cimport rand

cdef inline int int_max(int a, int b) nogil: return a if a >= b else b
cdef inline int int_min(int a, int b) nogil: return a if a <= b else b

cdef unsigned int _count_edges(UINT_t[:,:,:] X,
                 unsigned int i_start,
                 unsigned int i_end,
                 unsigned int j_start,
                 unsigned int j_end,
                 unsigned int num_z) nogil:
    cdef unsigned int count = 0
    cdef unsigned int i,j,z
    for i in range(i_start,i_end):
        for j in range(j_start,j_end):
            for z in range(num_z):
                if X[i,j,z]:
                    count += 1
    return count



# cdef compute_loglikelihoods(np.ndarray[ndim=2,dtype=UINT_t] X,
#                            unsigned int i_start,
#                            unsigned int i_end,
#                            unsigned int j_start,
#                            unsigned int j_end,
#                            np.ndarray[ndim=3,dtype=DTYPE_t] log_parts,
#                            np.ndarray[ndim=3,dtype=DTYPE_t] log_invparts,
#                            np.ndarray[ndim=3,dtype=DTYPE_t] out_map,
#                            unsigned int num_parts):
#     for i in range(i_end-i_start):
#         for j in range(j_end-j_start):
#             if X[i_start+i,j_start+j]:
#                 for k in range(num_parts):
#                     out_map[i_start,j_start,k] += log_parts[k,i,j]
#             else:
#                 for k in range(num_parts):
#                     out_map[i_start,j_start,k] += log_invparts[k,i,j]

cdef unsigned int _count_edges_mask(np.ndarray[ndim=3,dtype=UINT_t] X,
                      np.ndarray[ndim=2,dtype=UINT_t] M,
                 unsigned int i_start,
                 unsigned int i_end,
                 unsigned int j_start,
                 unsigned int j_end,
                 unsigned int num_z):
    cdef unsigned int count = 0
    cdef unsigned int i,j,z
    for i in range(i_start,i_end):
        for j in range(j_start,j_end):
            if M[i,j]:
                for z in range(num_z):
                    if X[i,j,z]:
                        count += 1
    return count


def code_parts(np.ndarray[ndim=3,dtype=UINT_t] X,
               np.ndarray[ndim=3,dtype=UINT_t] X_unspread,
               np.ndarray[ndim=4,dtype=np.float64_t] log_parts,
               np.ndarray[ndim=4,dtype=np.float64_t] log_invparts,
               int threshold, outer_frame=0, strides=1, int max_threshold=10000):
    """
    At each location of `X`, find the log probabilities for each part and location. Outputs these part assignments in the same data dimensions as `X`. Neighborhoods of `X` with edge counts lower than `threshold` are regarded as background and assigned zero.

    Parameters
    ----------
    X : ndarray[ndim=3,dtype=np.uint8]
        The first two dimensions of the array specify locations. The last one specifies a binary edge type. The value ``X[s,t,e]`` is 1 iff there is an edge of type `e` detected at location `(s,t)`.
    log_parts : ndarray[ndim=4]
        We have a Bernoulli mixture model defined over patches of the input image. The `log_parts` is a logarithm applied to the array of edge probability maps for each part. Array of shape `(K, S, T, E)`, where `K` is the number of mixture component, `S` and `T` the shape of the data, and `E` the number of edges. The value of ``log_parts[k,s,t,e]`` is the log probability of observing an edge `e` at location `(s,t)`, conditioned on the mixture component being `k`.
    log_invparts : ndarray[ndim=4]
        Preprocessed inverse of `log_parts`, i.e. ``log(1-exp(log_parts))``.
    threshold : int
        The least number of edges in a patch to reject the null background hypothesis.
    outer_frame : int
        Remove a frame of this thickness when checking the threshold. If the parts are 5 x 5, and this is set to 1, then only the center 3 x 3 is used to count edges when compared to the threshold. 
    strides : int
        When checking a part, these are the strides in both axes. For instance, if we have 9 x 9 parts and strides is set to 3, then it will only check 3 x 3 locations.
    
    Returns
    -------
    out_map : ndarray[ndim=3] 
        Array of shape `(S, T, K+1)`. There are two cases, either the third dimension is `(0, -inf, -inf, ...)`, when there are insufficient edges in the neighborhood of a location. Otherwise, `out_map[s,t,i+1]` is the log likelihood of part `i` at location `(s,t)`. Additionally, `out_map[s,t,0]` is equal to `-inf`.
    """

    cdef unsigned int num_parts = log_parts.shape[0]
    cdef unsigned int part_x_dim = log_parts.shape[1]
    cdef unsigned int part_y_dim = log_parts.shape[2]
    cdef unsigned int part_z_dim = log_parts.shape[3]
    cdef unsigned int X_x_dim = X.shape[0]
    cdef unsigned int X_y_dim = X.shape[1]
    cdef unsigned int X_z_dim = X.shape[2]
    cdef unsigned int new_x_dim = X_x_dim - part_x_dim + 1
    cdef unsigned int new_y_dim = X_y_dim - part_y_dim + 1
    cdef unsigned int i_start,j_start,i_end,j_end,count,i,j,z,k, cx0, cx1, cy0, cy1 
    cdef unsigned int i_frame = <unsigned int>outer_frame
    cdef int i_strides = <int>strides 
    cdef int i_offset = i_strides / 2
    cdef DTYPE_t NINF = DTYPE(-np.inf)
    # we have num_parts + 1 because we are also including some regions as being
    # thresholded due to there not being enough edges
    
    cdef np.ndarray[dtype=DTYPE_t, ndim=3] out_map = np.ones((new_x_dim,
                                                              new_y_dim,
                                                              num_parts+1),dtype=DTYPE) * DTYPE(-np.inf)
    cdef UINT_t[:,:,:] X_mv = X
    cdef UINT_t[:,:,:] X_unspread_mv = X_unspread

    cdef np.ndarray[dtype=DTYPE_t, ndim=4] part_logits = (log_parts - log_invparts).astype(DTYPE)

    cdef np.ndarray[dtype=DTYPE_t, ndim=1] constant_terms = np.apply_over_axes(np.sum, log_invparts[:,i_offset::i_strides,i_offset::i_strides].astype(DTYPE), [1, 2, 3]).ravel()
    cdef DTYPE_t[:,:,:,:] part_logits_mv = part_logits
    cdef DTYPE_t[:] constant_terms_mv = constant_terms

    cdef DTYPE_t[:,:,:] out_map_mv = out_map

    cdef np.ndarray[dtype=UINT32_t, ndim=2] _integral_counts = np.zeros((X_x_dim+1, X_y_dim+1), dtype=UINT32)
    cdef UINT32_t[:,:] integral_counts = _integral_counts

    # The first cell along the num_parts+1 axis contains a value that is either 0
    # if the area is deemed to have too few edges or min_val if there are sufficiently many
    # edges, min_val is just meant to be less than the value of the other cells
    # so when we pick the most likely part it won't be chosen

    with nogil:
        # Build integral image of edge counts
        # First, fill it with edge counts and accmulate across
        # one axis.
        for i in range(X_x_dim):
            for j in range(X_y_dim):
                count = 0
                for z in range(X_z_dim):
                    count += X_unspread_mv[i,j,z]
                integral_counts[1+i,1+j] = integral_counts[1+i,j] + count
        # Now accumulate the other axis
        for j in range(X_y_dim):
            for i in range(X_x_dim):
                integral_counts[1+i,1+j] += integral_counts[i,1+j]



        # Code parts
        for i_start in range(X_x_dim-part_x_dim+1):
            i_end = i_start + part_x_dim
            for j_start in range(X_y_dim-part_y_dim+1):
                j_end = j_start + part_y_dim
                #count = _count_edges(X_mv,i_start+i_frame,i_end-i_frame,j_start+i_frame,j_end-i_frame,X_z_dim)

                # Note, integral_counts is 1-based, to allow for a zero row/column at the zero:th index.
                cx0 = i_start+i_frame
                cx1 = i_end-i_frame
                cy0 = j_start+i_frame
                cy1 = j_end-i_frame
                count = integral_counts[cx1, cy1] - \
                        integral_counts[cx0, cy1] - \
                        integral_counts[cx1, cy0] + \
                        integral_counts[cx0, cy0]

                if threshold <= count <= max_threshold:
                    # Initialize to the constant term for each part and -inf to bkg
                    out_map_mv[i_start,j_start,0] = NINF 
                    for k in range(num_parts):
                        out_map_mv[i_start,j_start,1+k] = constant_terms_mv[k]

                    #for i in range(i_offset_x, part_x_dim, i_strides):
                        #for j in range(i_offset_y, part_y_dim, i_strides):
                    i = i_offset
                    while i < part_x_dim:
                        j = i_offset
                        while j < part_y_dim: 
                            for z in range(X_z_dim):
                                if X_mv[i_start+i,j_start+j,z]:
                                    for k in range(num_parts):
                                        out_map_mv[i_start,j_start,k+1] += part_logits_mv[k,i,j,z]

                            j += i_strides
                        i += i_strides
                else:
                    out_map_mv[i_start,j_start,0] = 0.0
                    # Rest is already -inf
                
    return out_map

def code_parts_new(np.ndarray[ndim=3,dtype=UINT_t] X,
                   np.ndarray[ndim=3,dtype=UINT_t] X_unspread,
                   np.ndarray[ndim=4,dtype=np.float64_t] log_parts,
                   np.ndarray[ndim=4,dtype=np.float64_t] log_invparts,
                   int threshold, outer_frame=0, int collapse=1):
    """
    At each location of `X`, find the log probabilities for each part and location. Outputs these part assignments in the same data dimensions as `X`. Neighborhoods of `X` with edge counts lower than `threshold` are regarded as background and assigned zero.

    Parameters
    ----------
    X : ndarray[ndim=3,dtype=np.uint8]
        The first two dimensions of the array specify locations. The last one specifies a binary edge type. The value ``X[s,t,e]`` is 1 iff there is an edge of type `e` detected at location `(s,t)`.
    log_parts : ndarray[ndim=4]
        We have a Bernoulli mixture model defined over patches of the input image. The `log_parts` is a logarithm applied to the array of edge probability maps for each part. Array of shape `(K, S, T, E)`, where `K` is the number of mixture component, `S` and `T` the shape of the data, and `E` the number of edges. The value of ``log_parts[k,s,t,e]`` is the log probability of observing an edge `e` at location `(s,t)`, conditioned on the mixture component being `k`.
    log_invparts : ndarray[ndim=4]
        Preprocessed inverse of `log_parts`, i.e. ``log(1-exp(log_parts))``.
    threshold : int
        The least number of edges in a patch to reject the null background hypothesis.
    outer_frame : int
        Remove a frame of this thickness when checking the threshold. If the parts are 5 x 5, and this is set to 1, then only the center 3 x 3 is used to count edges when compared to the threshold. 
    
    Returns
    -------
    out_map : ndarray[ndim=3] 
        Array of shape `(S, T, K+1)`. There are two cases, either the third dimension is `(0, -inf, -inf, ...)`, when there are insufficient edges in the neighborhood of a location. Otherwise, `out_map[s,t,i+1]` is the log likelihood of part `i` at location `(s,t)`. Additionally, `out_map[s,t,0]` is equal to `-inf`.
    """

    cdef unsigned int num_parts = log_parts.shape[0]
    cdef unsigned int part_x_dim = log_parts.shape[1]
    cdef unsigned int part_y_dim = log_parts.shape[2]
    cdef unsigned int part_z_dim = log_parts.shape[3]
    cdef unsigned int X_x_dim = X.shape[0]
    cdef unsigned int X_y_dim = X.shape[1]
    cdef unsigned int X_z_dim = X.shape[2]
    cdef unsigned int new_x_dim = X_x_dim - part_x_dim + 1
    cdef unsigned int new_y_dim = X_y_dim - part_y_dim + 1
    cdef unsigned int i_start,j_start,i_end,j_end,count,i,j,z,k, cx0, cx1, cy0, cy1 
    cdef unsigned int i_frame = <unsigned int>outer_frame
    cdef DTYPE_t NINF = DTYPE(-np.inf)
    # we have num_parts + 1 because we are also including some regions as being
    # thresholded due to there not being enough edges
    
    cdef np.ndarray[dtype=np.int32_t, ndim=2] out_map = -np.ones((new_x_dim,
                                                                 new_y_dim), dtype=np.int32)
    cdef np.ndarray[dtype=DTYPE_t, ndim=1] vs = np.ones(num_parts, dtype=DTYPE)
    cdef DTYPE_t[:] vs_mv = vs

    cdef UINT_t[:,:,:] X_mv = X
    cdef UINT_t[:,:,:] X_unspread_mv = X_unspread

    cdef np.ndarray[dtype=DTYPE_t, ndim=4] part_logits = np.rollaxis((log_parts - log_invparts).astype(DTYPE), 0, 4).copy()

    cdef np.ndarray[dtype=DTYPE_t, ndim=1] constant_terms = np.apply_over_axes(np.sum, log_invparts.astype(DTYPE), [1, 2, 3]).ravel()
    cdef DTYPE_t[:,:,:,:] part_logits_mv = part_logits
    cdef DTYPE_t[:] constant_terms_mv = constant_terms

    cdef np.int32_t[:,:] out_map_mv = out_map

    cdef np.ndarray[dtype=UINT32_t, ndim=2] _integral_counts = np.zeros((X_x_dim+1, X_y_dim+1), dtype=UINT32)
    cdef UINT32_t[:,:] integral_counts = _integral_counts

    cdef DTYPE_t max_value, v
    cdef int max_index 

    # The first cell along the num_parts+1 axis contains a value that is either 0
    # if the area is deemed to have too few edges or min_val if there are sufficiently many
    # edges, min_val is just meant to be less than the value of the other cells
    # so when we pick the most likely part it won't be chosen

    # Build integral image of edge counts
    # First, fill it with edge counts and accmulate across
    # one axis.
    for i in range(X_x_dim):
        for j in range(X_y_dim):
            count = 0
            for z in range(X_z_dim):
                count += X_unspread_mv[i,j,z]
            integral_counts[1+i,1+j] = integral_counts[1+i,j] + count
    # Now accumulate the other axis
    for j in range(X_y_dim):
        for i in range(X_x_dim):
            integral_counts[1+i,1+j] += integral_counts[i,1+j]

    # Code parts
    for i_start in range(X_x_dim-part_x_dim+1):
        i_end = i_start + part_x_dim
        for j_start in range(X_y_dim-part_y_dim+1):
            j_end = j_start + part_y_dim

            # Note, integral_counts is 1-based, to allow for a zero row/column at the zero:th index.
            cx0 = i_start+i_frame
            cx1 = i_end-i_frame
            cy0 = j_start+i_frame
            cy1 = j_end-i_frame
            count = integral_counts[cx1, cy1] - \
                    integral_counts[cx0, cy1] - \
                    integral_counts[cx1, cy0] + \
                    integral_counts[cx0, cy0]

            if threshold <= count:
                vs[:] = constant_terms
                with nogil:
                    for i in range(part_x_dim):
                        for j in range(part_y_dim):
                            for z in range(X_z_dim):
                                if X_mv[i_start+i,j_start+j,z]:
                                    for k in range(num_parts):
                                        vs_mv[k] += part_logits_mv[i,j,z,k]

                out_map_mv[i_start,j_start] = vs.argmax() / collapse

    return out_map

def code_parts_mmm(np.ndarray[ndim=3,dtype=UINT_t] X,
                   np.ndarray[ndim=3,dtype=UINT_t] X_unspread,
                   np.ndarray[ndim=4,dtype=np.float64_t] log_parts,
                   np.ndarray[ndim=4,dtype=np.float64_t] log_invparts,
                   np.ndarray[ndim=1,dtype=np.int64_t] part_to_feature,
                   int threshold, outer_frame=0, 
                   int stride=1):
    """
    At each location of `X`, find the log probabilities for each part and location. Outputs these part assignments in the same data dimensions as `X`. Neighborhoods of `X` with edge counts lower than `threshold` are regarded as background and assigned zero.

    Parameters
    ----------
    X : ndarray[ndim=3,dtype=np.uint8]
        The first two dimensions of the array specify locations. The last one specifies a binary edge type. The value ``X[s,t,e]`` is 1 iff there is an edge of type `e` detected at location `(s,t)`.
    log_parts : ndarray[ndim=4]
        We have a Bernoulli mixture model defined over patches of the input image. The `log_parts` is a logarithm applied to the array of edge probability maps for each part. Array of shape `(K, S, T, E)`, where `K` is the number of mixture component, `S` and `T` the shape of the data, and `E` the number of edges. The value of ``log_parts[k,s,t,e]`` is the log probability of observing an edge `e` at location `(s,t)`, conditioned on the mixture component being `k`.
    log_invparts : ndarray[ndim=4]
        Preprocessed inverse of `log_parts`, i.e. ``log(1-exp(log_parts))``.
    threshold : int
        The least number of edges in a patch to reject the null background hypothesis.
    outer_frame : int
        Remove a frame of this thickness when checking the threshold. If the parts are 5 x 5, and this is set to 1, then only the center 3 x 3 is used to count edges when compared to the threshold. 
    
    Returns
    -------
    out_map : ndarray[ndim=3] 
        Array of shape `(S, T, K+1)`. There are two cases, either the third dimension is `(0, -inf, -inf, ...)`, when there are insufficient edges in the neighborhood of a location. Otherwise, `out_map[s,t,i+1]` is the log likelihood of part `i` at location `(s,t)`. Additionally, `out_map[s,t,0]` is equal to `-inf`.
    """

    cdef unsigned int num_parts = log_parts.shape[0]
    cdef unsigned int part_x_dim = log_parts.shape[1]
    cdef unsigned int part_y_dim = log_parts.shape[2]
    cdef unsigned int part_z_dim = log_parts.shape[3]
    cdef unsigned int X_x_dim = X.shape[0]
    cdef unsigned int X_y_dim = X.shape[1]
    cdef unsigned int X_z_dim = X.shape[2]
    cdef unsigned int new_x_dim = X_x_dim - part_x_dim + 1
    cdef unsigned int new_y_dim = X_y_dim - part_y_dim + 1
    cdef unsigned int i_start,j_start,i_end,j_end,count,i,j,z,k, cx0, cx1, cy0, cy1 
    cdef unsigned int i_frame = <unsigned int>outer_frame
    cdef DTYPE_t NINF = DTYPE(-np.inf)
    # we have num_parts + 1 because we are also including some regions as being
    # thresholded due to there not being enough edges
    
    cdef np.ndarray[dtype=np.int32_t, ndim=2] out_map = -np.ones((new_x_dim,
                                                                 new_y_dim), dtype=np.int32)
    cdef np.ndarray[dtype=DTYPE_t, ndim=1] vs = np.ones(num_parts, dtype=DTYPE)
    cdef DTYPE_t[:] vs_mv = vs

    cdef np.int64_t[:] part_to_feature_mv = part_to_feature

    #cdef np.ndarray[dtype=DTYPE_t, ndim=1] vs_alt = np.ones(num_parts, dtype=DTYPE)
    #cdef DTYPE_t[:] vs_alt_mv = vs_alt

    parts = np.exp(log_parts)
    #parts_alt = np.tile(np.apply_over_axes(np.mean, parts, [1, 2]), (part_x_dim, part_y_dim, 1))
    #parts_alt = np.tile(np.apply_over_axes(np.mean, parts, [1, 2, 3]), (part_x_dim, part_y_dim, part_z_dim))

    #cdef np.ndarray[dtype=DTYPE_t, ndim=4] part_logits_alt = np.rollaxis((np.log(parts_alt) - np.log(1 - parts_alt)).astype(DTYPE), 0, 4).copy()
    #cdef np.ndarray[dtype=DTYPE_t, ndim=1] constant_terms_alt = np.apply_over_axes(np.sum, np.log(1 - parts_alt).astype(DTYPE), [1, 2, 3]).ravel()

    #cdef DTYPE_t[:,:,:,:] part_logits_alt_mv = part_logits_alt
    #cdef DTYPE_t[:] constant_terms_alt_mv = constant_terms_alt

    cdef UINT_t[:,:,:] X_mv = X
    cdef UINT_t[:,:,:] X_unspread_mv = X_unspread

    cdef np.ndarray[dtype=DTYPE_t, ndim=4] part_logits = np.rollaxis((log_parts - log_invparts).astype(DTYPE), 0, 4).copy()

    cdef np.ndarray[dtype=DTYPE_t, ndim=1] constant_terms = np.apply_over_axes(np.sum, log_invparts.astype(DTYPE), [1, 2, 3]).ravel()
    cdef DTYPE_t[:,:,:,:] part_logits_mv = part_logits
    cdef DTYPE_t[:] constant_terms_mv = constant_terms

    cdef np.int32_t[:,:] out_map_mv = out_map

    cdef np.ndarray[dtype=UINT32_t, ndim=2] _integral_counts = np.zeros((X_x_dim+1, X_y_dim+1), dtype=UINT32)
    cdef UINT32_t[:,:] integral_counts = _integral_counts

    cdef DTYPE_t max_value, v
    cdef int max_index 

    # The first cell along the num_parts+1 axis contains a value that is either 0
    # if the area is deemed to have too few edges or min_val if there are sufficiently many
    # edges, min_val is just meant to be less than the value of the other cells
    # so when we pick the most likely part it won't be chosen

    # Build integral image of edge counts
    # First, fill it with edge counts and accmulate across
    # one axis.
    for i in range(X_x_dim):
        for j in range(X_y_dim):
            count = 0
            for z in range(X_z_dim):
                count += X_unspread_mv[i,j,z]
            integral_counts[1+i,1+j] = integral_counts[1+i,j] + count
    # Now accumulate the other axis
    for j in range(X_y_dim):
        for i in range(X_x_dim):
            integral_counts[1+i,1+j] += integral_counts[i,1+j]

    # Code parts
    for i_start in range(0, X_x_dim-part_x_dim+1, stride):
        i_end = i_start + part_x_dim
        for j_start in range(0, X_y_dim-part_y_dim+1, stride):
            j_end = j_start + part_y_dim

            # Note, integral_counts is 1-based, to allow for a zero row/column at the zero:th index.
            cx0 = i_start+i_frame
            cx1 = i_end-i_frame
            cy0 = j_start+i_frame
            cy1 = j_end-i_frame
            count = integral_counts[cx1, cy1] - \
                    integral_counts[cx0, cy1] - \
                    integral_counts[cx1, cy0] + \
                    integral_counts[cx0, cy0]

            if threshold <= count:
                vs[:] = constant_terms
                #vs_alt[:] = constant_terms_alt
                with nogil:
                    for i in range(part_x_dim):
                        for j in range(part_y_dim):
                            for z in range(X_z_dim):
                                if X_mv[i_start+i,j_start+j,z]:
                                    for k in range(num_parts):
                                        vs_mv[k] += part_logits_mv[i,j,z,k]
                                        #vs_alt_mv[k] += part_logits_alt_mv[i,j,z,k]

                max_index = vs.argmax()
                #if vs_mv[max_index] - vs_alt_mv[max_index] >= accept_threshold:
                out_map_mv[i_start,j_start] = part_to_feature_mv[max_index]
                
                #out_map_mv[i_start,j_start] = vs.argmax() / collapse

    return out_map


def code_parts_mmm_adaptive_EXPERIMENTAL(np.ndarray[ndim=3,dtype=UINT_t] X,
                   np.ndarray[ndim=3,dtype=UINT_t] X_unspread,
                   np.ndarray[ndim=4,dtype=np.float64_t] log_parts,
                   np.ndarray[ndim=4,dtype=np.float64_t] log_invparts,
                   int threshold, outer_frame=0, int collapse=1, accept_threshold=1.0):
    """
    At each location of `X`, find the log probabilities for each part and location. Outputs these part assignments in the same data dimensions as `X`. Neighborhoods of `X` with edge counts lower than `threshold` are regarded as background and assigned zero.

    Parameters
    ----------
    X : ndarray[ndim=3,dtype=np.uint8]
        The first two dimensions of the array specify locations. The last one specifies a binary edge type. The value ``X[s,t,e]`` is 1 iff there is an edge of type `e` detected at location `(s,t)`.
    log_parts : ndarray[ndim=4]
        We have a Bernoulli mixture model defined over patches of the input image. The `log_parts` is a logarithm applied to the array of edge probability maps for each part. Array of shape `(K, S, T, E)`, where `K` is the number of mixture component, `S` and `T` the shape of the data, and `E` the number of edges. The value of ``log_parts[k,s,t,e]`` is the log probability of observing an edge `e` at location `(s,t)`, conditioned on the mixture component being `k`.
    log_invparts : ndarray[ndim=4]
        Preprocessed inverse of `log_parts`, i.e. ``log(1-exp(log_parts))``.
    threshold : int
        The least number of edges in a patch to reject the null background hypothesis.
    outer_frame : int
        Remove a frame of this thickness when checking the threshold. If the parts are 5 x 5, and this is set to 1, then only the center 3 x 3 is used to count edges when compared to the threshold. 
    
    Returns
    -------
    out_map : ndarray[ndim=3] 
        Array of shape `(S, T, K+1)`. There are two cases, either the third dimension is `(0, -inf, -inf, ...)`, when there are insufficient edges in the neighborhood of a location. Otherwise, `out_map[s,t,i+1]` is the log likelihood of part `i` at location `(s,t)`. Additionally, `out_map[s,t,0]` is equal to `-inf`.
    """

    cdef unsigned int num_parts = log_parts.shape[0]
    cdef unsigned int part_x_dim = log_parts.shape[1]
    cdef unsigned int part_y_dim = log_parts.shape[2]
    cdef unsigned int part_z_dim = log_parts.shape[3]
    cdef unsigned int X_x_dim = X.shape[0]
    cdef unsigned int X_y_dim = X.shape[1]
    cdef unsigned int X_z_dim = X.shape[2]
    cdef unsigned int new_x_dim = X_x_dim - part_x_dim + 1
    cdef unsigned int new_y_dim = X_y_dim - part_y_dim + 1
    cdef unsigned int i_start,j_start,i_end,j_end,count,i,j,z,k, cx0, cx1, cy0, cy1 
    cdef unsigned int i_frame = <unsigned int>outer_frame
    cdef DTYPE_t NINF = DTYPE(-np.inf)
    # we have num_parts + 1 because we are also including some regions as being
    # thresholded due to there not being enough edges
    
    cdef np.ndarray[dtype=np.int32_t, ndim=2] out_map = -np.ones((new_x_dim,
                                                                 new_y_dim), dtype=np.int32)
    cdef np.ndarray[dtype=DTYPE_t, ndim=1] vs = np.ones(num_parts, dtype=DTYPE)
    cdef DTYPE_t[:] vs_mv = vs

    cdef DTYPE_t vs_alt = 0.0

    parts = np.exp(log_parts)
    parts_alt = np.tile(np.apply_over_axes(np.mean, parts, [1, 2]), (part_x_dim, part_y_dim, 1))
    #parts_alt = np.tile(np.apply_over_axes(np.mean, parts, [1, 2, 3]), (part_x_dim, part_y_dim, part_z_dim))

    cdef DTYPE_t constant_terms_alt = 0.0

    cdef np.ndarray[dtype=DTYPE_t, ndim=1] bkg = np.zeros(X_z_dim)
    cdef np.ndarray[dtype=DTYPE_t, ndim=1] log_bkg = np.zeros(X_z_dim)
    cdef np.ndarray[dtype=DTYPE_t, ndim=1] log_invbkg = np.zeros(X_z_dim)
    cdef np.ndarray[dtype=DTYPE_t, ndim=1] part_logits_alt = np.zeros(X_z_dim)

    cdef DTYPE_t[:] part_logits_alt_mv = part_logits_alt
    #cdef DTYPE_t[:] c= constant_terms_alt

    cdef UINT_t[:,:,:] X_mv = X
    cdef UINT_t[:,:,:] X_unspread_mv = X_unspread

    #cdedef 
    cdef np.ndarray[dtype=DTYPE_t, ndim=4] part_logits = np.rollaxis((log_parts - log_invparts).astype(DTYPE), 0, 4).copy()

    cdef np.ndarray[dtype=DTYPE_t, ndim=1] constant_terms = np.apply_over_axes(np.sum, log_invparts.astype(DTYPE), [1, 2, 3]).ravel()
    cdef DTYPE_t[:,:,:,:] part_logits_mv = part_logits
    cdef DTYPE_t[:] constant_terms_mv = constant_terms

    cdef np.int32_t[:,:] out_map_mv = out_map

    cdef np.ndarray[dtype=UINT32_t, ndim=2] _integral_counts = np.zeros((X_x_dim+1, X_y_dim+1), dtype=UINT32)
    cdef UINT32_t[:,:] integral_counts = _integral_counts

    cdef DTYPE_t max_value, v
    cdef int max_index 


    import amitgroup as ag
    S = 10
    hS = S // 2
    edge_count = X.astype(np.int64).cumsum(0).cumsum(1).astype(np.float64) / S**2
    padded = ag.util.border_value_pad_upper(edge_count, hS + 1)
    bkg_models = padded[S:S+X_x_dim,S:S+X_y_dim] - padded[0:0+X_x_dim,S:S+X_y_dim] - padded[S:S+X_x_dim,0:0+X_y_dim] + padded[0:0+X_x_dim,0:0+X_y_dim]
    bkg_models = np.clip(bkg_models, 0.025, 1-0.025)

    # The first cell along the num_parts+1 axis contains a value that is either 0
    # if the area is deemed to have too few edges or min_val if there are sufficiently many
    # edges, min_val is just meant to be less than the value of the other cells
    # so when we pick the most likely part it won't be chosen

    # Build integral image of edge counts
    # First, fill it with edge counts and accmulate across
    # one axis.
    for i in range(X_x_dim):
        for j in range(X_y_dim):
            count = 0
            for z in range(X_z_dim):
                count += X_unspread_mv[i,j,z]
            integral_counts[1+i,1+j] = integral_counts[1+i,j] + count
    # Now accumulate the other axis
    for j in range(X_y_dim):
        for i in range(X_x_dim):
            integral_counts[1+i,1+j] += integral_counts[i,1+j]

    # Code parts
    for i_start in range(X_x_dim-part_x_dim+1):
        i_end = i_start + part_x_dim
        for j_start in range(X_y_dim-part_y_dim+1):
            j_end = j_start + part_y_dim

            # Note, integral_counts is 1-based, to allow for a zero row/column at the zero:th index.
            cx0 = i_start+i_frame
            cx1 = i_end-i_frame
            cy0 = j_start+i_frame
            cy1 = j_end-i_frame
            count = integral_counts[cx1, cy1] - \
                    integral_counts[cx0, cy1] - \
                    integral_counts[cx1, cy0] + \
                    integral_counts[cx0, cy0]

            # Set background
            bkg[:] = bkg_models[i,j]
            log_bkg[:] = np.log(bkg)
            log_invbkg[:] = np.log(1 - bkg)
            
            constant_terms_alt = log_invbkg.sum()
            part_logits_alt[:] = log_bkg - log_invbkg

            if threshold <= count:
                vs[:] = constant_terms
                vs_alt = constant_terms_alt
                with nogil:
                    for i in range(part_x_dim):
                        for j in range(part_y_dim):
                            for z in range(X_z_dim):
                                if X_mv[i_start+i,j_start+j,z]:
                                    vs_alt += part_logits_alt_mv[z]
                                    for k in range(num_parts):
                                        vs_mv[k] += part_logits_mv[i,j,z,k]

                max_index = vs.argmax()
                if vs_mv[max_index] - vs_alt >= accept_threshold:
                    out_map_mv[i_start,j_start] = max_index / collapse
                
                #out_map_mv[i_start,j_start] = vs.argmax() / collapse

    return out_map

def subsample_offset_shape(shape, size):
    return [int(shape[i]%size[i]/2 + size[i]/2)  for i in range(2)]

def extract_parts(edges, unspread_edges, log_parts, log_invparts, int threshold, 
                  outer_frame=0, spread_radii=(4, 4), subsample_size=(4, 4), 
                  part_to_feature=None, int stride=1,
                  between_feature_spreading=None):
    if part_to_feature is None:
        part_to_feature = np.arange(log_parts.shape[0])

    if between_feature_spreading is None:
        between_feature_spreading = np.arange(log_parts.shape[0], dtype=np.int32)[...,np.newaxis]

    cdef:
        # One above the top index
        int num_feats = part_to_feature.max()+1
        np.ndarray[np.int32_t,ndim=2] parts = code_parts_mmm(edges, unspread_edges, log_parts, log_invparts, part_to_feature, threshold, outer_frame=outer_frame, stride=stride)
        np.ndarray[np.uint8_t,ndim=3] feats = np.zeros((parts.shape[0]//subsample_size[0],
                                                        parts.shape[1]//subsample_size[1],
                                                        num_feats),
                                                        dtype=np.uint8)

    offset = subsample_offset_shape((parts.shape[0], parts.shape[1]), subsample_size)

    cdef:
        np.int32_t[:,:] parts_mv = parts
        np.uint8_t[:,:,:] feats_mv = feats
        np.int32_t[:,:] between_feature_spreading_mv = between_feature_spreading

        int bfs_mult = between_feature_spreading.shape[1]
  
        int spread_radii0 = spread_radii[0]
        int spread_radii1 = spread_radii[1]
        int subsample_size0 = subsample_size[0]
        int subsample_size1 = subsample_size[1]
        int feats_dim0 = feats.shape[0]
        int feats_dim1 = feats.shape[1]
        int parts_dim0 = parts.shape[0]
        int parts_dim1 = parts.shape[1]
        int offset0 = offset[0]
        int offset1 = offset[1]
        int p, x, y, i, j, i0, j0, b

    with nogil:
        for i in range(feats_dim0):
            for j in range(feats_dim1):
                x = offset0 + i*subsample_size0 
                y = offset1 + j*subsample_size1
                for i0 in range(int_max(x - spread_radii0, 0), int_min(x + spread_radii0+1, parts_dim0)):
                    for j0 in range(int_max(y - spread_radii1, 0), int_min(y + spread_radii1+1, parts_dim1)):
                        p = parts_mv[i0,j0]
                        if p != -1:
                            #feats_mv[i,j,p] = 1
                            for b in range(bfs_mult):
                                feats_mv[i,j,between_feature_spreading_mv[p,b]] = 1

    return feats 

def extract_parts_adaptive_EXPERIMENTAL(edges, unspread_edges, log_parts, log_invparts, int threshold, outer_frame=0, spread_radii=(4, 4), subsample_size=(4, 4), int collapse=1, accept_threshold=0.0, TMP_each=1):
    cdef:
        int num_feats = log_parts.shape[0]//collapse
        np.ndarray[np.int32_t,ndim=2] parts = code_parts_mmm(edges, unspread_edges, log_parts, log_invparts, threshold, outer_frame=outer_frame, collapse=collapse, accept_threshold=accept_threshold)
        np.ndarray[np.uint8_t,ndim=3] feats = np.zeros((parts.shape[0]//subsample_size[0],
                                                        parts.shape[1]//subsample_size[1],
                                                        num_feats * TMP_each),
                                                        dtype=np.uint8)


    if TMP_each > 1:
        coded_parts = parts[parts != -1]
        coded_parts *= TMP_each 
        coded_parts += np.random.randint(TMP_each, size=coded_parts.size)
        parts[parts != -1] = coded_parts
        num_feats *= TMP_each

    offset = subsample_offset_shape((parts.shape[0], parts.shape[1]), subsample_size)


    cdef:
        np.int32_t[:,:] parts_mv = parts
        np.uint8_t[:,:,:] feats_mv = feats

        int spread_radii0 = spread_radii[0]
        int spread_radii1 = spread_radii[1]
        int subsample_size0 = subsample_size[0]
        int subsample_size1 = subsample_size[1]
        int feats_dim0 = feats.shape[0]
        int feats_dim1 = feats.shape[1]
        int parts_dim0 = parts.shape[0]
        int parts_dim1 = parts.shape[1]
        int offset0 = offset[0]
        int offset1 = offset[1]
        int p, x, y, i, j, i0, j0

    with nogil:
        for i in range(feats_dim0):
            for j in range(feats_dim1):
                x = offset0 + i*subsample_size0 
                y = offset1 + j*subsample_size1
                for i0 in range(int_max(x - spread_radii0, 0), int_min(x + spread_radii0+1, parts_dim0)):
                    for j0 in range(int_max(y - spread_radii1, 0), int_min(y + spread_radii1+1, parts_dim1)):
                        p = parts_mv[i0,j0]
                        if p != -1:
                            feats_mv[i,j,p] = 1

    return feats 

if 0:
    def code_polarity_parts(np.ndarray[ndim=4,dtype=UINT_t] X,
                            np.ndarray[ndim=4,dtype=UINT_t] X_unspread,
                            np.ndarray[ndim=5,dtype=np.float64_t] log_parts,
                            np.ndarray[ndim=5,dtype=np.float64_t] log_invparts,
                            int threshold, outer_frame=0, strides=1, int max_threshold=10000):

        # Must have two polarities on these axes
        assert log_parts.shape[1] == 2
        assert X.shape[0] == 2
        cdef unsigned int num_parts = log_parts.shape[0]
        cdef unsigned int part_x_dim = log_parts.shape[2]
        cdef unsigned int part_y_dim = log_parts.shape[3]
        cdef unsigned int part_z_dim = log_parts.shape[4]
        cdef unsigned int X_x_dim = X.shape[1]
        cdef unsigned int X_y_dim = X.shape[2]
        cdef unsigned int X_z_dim = X.shape[3]
        cdef unsigned int new_x_dim = X_x_dim - part_x_dim + 1
        cdef unsigned int new_y_dim = X_y_dim - part_y_dim + 1
        cdef unsigned int i_start,j_start,i_end,j_end,count,i,j,z,k, cx0, cx1, cy0, cy1 
        cdef unsigned int i_frame = <unsigned int>outer_frame
        cdef int i_strides = <int>strides 
        cdef int i_offset = i_strides / 2
        cdef DTYPE_t NINF = DTYPE(-np.inf)
        # we have num_parts + 1 because we are also including some regions as being
        # thresholded due to there not being enough edges
        
        cdef np.ndarray[dtype=DTYPE_t, ndim=3] out_map = np.ones((new_x_dim,
                                                                  new_y_dim,
                                                                  num_parts+1),dtype=DTYPE) * DTYPE(-np.inf)
        cdef UINT_t[:,:,:] X_mv = X
        cdef UINT_t[:,:,:] X_unspread_mv = X_unspread

        cdef np.ndarray[dtype=DTYPE_t, ndim=4] part_logits = (log_parts - log_invparts).astype(DTYPE)

        cdef np.ndarray[dtype=DTYPE_t, ndim=1] constant_terms = np.apply_over_axes(np.sum, log_invparts[:,i_offset::i_strides,i_offset::i_strides].astype(DTYPE), [1, 2, 3]).ravel()
        cdef DTYPE_t[:,:,:,:] part_logits_mv = part_logits
        cdef DTYPE_t[:] constant_terms_mv = constant_terms

        cdef DTYPE_t[:,:,:] out_map_mv = out_map

        cdef np.ndarray[dtype=UINT32_t, ndim=2] _integral_counts = np.zeros((X_x_dim+1, X_y_dim+1), dtype=UINT32)
        cdef UINT32_t[:,:] integral_counts = _integral_counts

        # The first cell along the num_parts+1 axis contains a value that is either 0
        # if the area is deemed to have too few edges or min_val if there are sufficiently many
        # edges, min_val is just meant to be less than the value of the other cells
        # so when we pick the most likely part it won't be chosen

        with nogil:
            # Build integral image of edge counts
            # First, fill it with edge counts and accmulate across
            # one axis.
            for i in range(X_x_dim):
                for j in range(X_y_dim):
                    count = 0
                    for z in range(X_z_dim):
                        count += X_unspread_mv[i,j,z]
                    integral_counts[1+i,1+j] = integral_counts[1+i,j] + count
            # Now accumulate the other axis
            for j in range(X_y_dim):
                for i in range(X_x_dim):
                    integral_counts[1+i,1+j] += integral_counts[i,1+j]



            # Code parts
            for i_start in range(X_x_dim-part_x_dim+1):
                i_end = i_start + part_x_dim
                for j_start in range(X_y_dim-part_y_dim+1):
                    j_end = j_start + part_y_dim
                    #count = _count_edges(X_mv,i_start+i_frame,i_end-i_frame,j_start+i_frame,j_end-i_frame,X_z_dim)

                    # Note, integral_counts is 1-based, to allow for a zero row/column at the zero:th index.
                    cx0 = i_start+i_frame
                    cx1 = i_end-i_frame
                    cy0 = j_start+i_frame
                    cy1 = j_end-i_frame
                    count = integral_counts[cx1, cy1] - \
                            integral_counts[cx0, cy1] - \
                            integral_counts[cx1, cy0] + \
                            integral_counts[cx0, cy0]

                    if threshold <= count <= max_threshold:
                        # Initialize to the constant term for each part and -inf to bkg
                        out_map_mv[i_start,j_start,0] = NINF 
                        for k in range(num_parts):
                            out_map_mv[i_start,j_start,1+k] = constant_terms_mv[k]

                        #for i in range(i_offset_x, part_x_dim, i_strides):
                            #for j in range(i_offset_y, part_y_dim, i_strides):
                        i = i_offset
                        while i < part_x_dim:
                            j = i_offset
                            while j < part_y_dim: 
                                for z in range(X_z_dim):
                                    if X_mv[i_start+i,j_start+j,z]:
                                        for k in range(num_parts):
                                            out_map_mv[i_start,j_start,k+1] += part_logits_mv[k,i,j,z]

                                j += i_strides
                            i += i_strides
                    else:
                        out_map_mv[i_start,j_start,0] = 0.0
                        # Rest is already -inf
                    
        return out_map


def code_parts_INDICES(np.ndarray[ndim=3,dtype=UINT_t] X,
               np.ndarray[ndim=3,dtype=UINT_t] X_unspread,
               np.ndarray[ndim=4,dtype=np.float64_t] log_parts,
               np.ndarray[ndim=4,dtype=np.float64_t] log_invparts,
               np.ndarray[ndim=3,dtype=UINT32_t] indices,
               int threshold, outer_frame=0, strides=1, int max_threshold=10000):
    """
    At each location of `X`, find the log probabilities for each part and location. Outputs these part assignments in the same data dimensions as `X`. Neighborhoods of `X` with edge counts lower than `threshold` are regarded as background and assigned zero.

    Parameters
    ----------
    X : ndarray[ndim=3,dtype=np.uint8]
        The first two dimensions of the array specify locations. The last one specifies a binary edge type. The value ``X[s,t,e]`` is 1 iff there is an edge of type `e` detected at location `(s,t)`.
    log_parts : ndarray[ndim=4]
        We have a Bernoulli mixture model defined over patches of the input image. The `log_parts` is a logarithm applied to the array of edge probability maps for each part. Array of shape `(K, S, T, E)`, where `K` is the number of mixture component, `S` and `T` the shape of the data, and `E` the number of edges. The value of ``log_parts[k,s,t,e]`` is the log probability of observing an edge `e` at location `(s,t)`, conditioned on the mixture component being `k`.
    log_invparts : ndarray[ndim=4]
        Preprocessed inverse of `log_parts`, i.e. ``log(1-exp(log_parts))``.
    threshold : int
        The least number of edges in a patch to reject the null background hypothesis.
    outer_frame : int
        Remove a frame of this thickness when checking the threshold. If the parts are 5 x 5, and this is set to 1, then only the center 3 x 3 is used to count edges when compared to the threshold. 
    strides : int
        When checking a part, these are the strides in both axes. For instance, if we have 9 x 9 parts and strides is set to 3, then it will only check 3 x 3 locations.
    
    Returns
    -------
    out_map : ndarray[ndim=3] 
        Array of shape `(S, T, K+1)`. There are two cases, either the third dimension is `(0, -inf, -inf, ...)`, when there are insufficient edges in the neighborhood of a location. Otherwise, `out_map[s,t,i+1]` is the log likelihood of part `i` at location `(s,t)`. Additionally, `out_map[s,t,0]` is equal to `-inf`.
    """

    cdef unsigned int num_indices = indices.shape[1]
    cdef unsigned int num_parts = log_parts.shape[0]
    cdef unsigned int part_x_dim = log_parts.shape[1]
    cdef unsigned int part_y_dim = log_parts.shape[2]
    cdef unsigned int part_z_dim = log_parts.shape[3]
    cdef unsigned int X_x_dim = X.shape[0]
    cdef unsigned int X_y_dim = X.shape[1]
    cdef unsigned int X_z_dim = X.shape[2]
    cdef unsigned int new_x_dim = X_x_dim - part_x_dim + 1
    cdef unsigned int new_y_dim = X_y_dim - part_y_dim + 1
    cdef unsigned int i_start,j_start,i_end,j_end,count,i,j,index,z,k, cx0, cx1, cy0, cy1 
    cdef unsigned int i_frame = <unsigned int>outer_frame
    cdef int i_strides = <int>strides 
    cdef int i_offset = i_strides / 2
    cdef DTYPE_t NINF = DTYPE(-np.inf)
    cdef UINT_t b
    # we have num_parts + 1 because we are also including some regions as being
    # thresholded due to there not being enough edges
    
    cdef np.ndarray[dtype=DTYPE_t, ndim=3] out_map = np.ones((new_x_dim,
                                                              new_y_dim,
                                                              num_parts+1),dtype=DTYPE) * DTYPE(-np.inf)
    cdef UINT_t[:,:,:] X_mv = X
    cdef UINT_t[:,:,:] X_unspread_mv = X_unspread
    cdef UINT32_t[:,:,:] indices_mv = indices

    cdef np.float64_t[:,:,:,:] log_parts_mv = log_parts
    cdef np.float64_t[:,:,:,:] log_invparts_mv = log_invparts

    cdef np.ndarray[dtype=DTYPE_t, ndim=4] part_logits = (log_parts - log_invparts).astype(DTYPE)

    cdef np.ndarray[dtype=DTYPE_t, ndim=1] constant_terms = np.apply_over_axes(np.sum, log_invparts[:,i_offset::i_strides,i_offset::i_strides].astype(DTYPE), [1, 2, 3]).ravel()
    cdef DTYPE_t[:,:,:,:] part_logits_mv = part_logits
    cdef DTYPE_t[:] constant_terms_mv = constant_terms

    cdef DTYPE_t[:,:,:] out_map_mv = out_map

    cdef np.ndarray[dtype=UINT32_t, ndim=2] _integral_counts = np.zeros((X_x_dim+1, X_y_dim+1), dtype=UINT32)
    cdef UINT32_t[:,:] integral_counts = _integral_counts

    # The first cell along the num_parts+1 axis contains a value that is either 0
    # if the area is deemed to have too few edges or min_val if there are sufficiently many
    # edges, min_val is just meant to be less than the value of the other cells
    # so when we pick the most likely part it won't be chosen

    with nogil:
        # Build integral image of edge counts
        # First, fill it with edge counts and accmulate across
        # one axis.
        for i in range(X_x_dim):
            for j in range(X_y_dim):
                count = 0
                for z in range(X_z_dim):
                    count += X_unspread_mv[i,j,z]
                integral_counts[1+i,1+j] = integral_counts[1+i,j] + count
        # Now accumulate the other axis
        for j in range(X_y_dim):
            for i in range(X_x_dim):
                integral_counts[1+i,1+j] += integral_counts[i,1+j]



        # Code parts
        for i_start in range(X_x_dim-part_x_dim+1):
            i_end = i_start + part_x_dim
            for j_start in range(X_y_dim-part_y_dim+1):
                j_end = j_start + part_y_dim
                #count = _count_edges(X_mv,i_start+i_frame,i_end-i_frame,j_start+i_frame,j_end-i_frame,X_z_dim)

                # Note, integral_counts is 1-based, to allow for a zero row/column at the zero:th index.
                cx0 = i_start+i_frame
                cx1 = i_end-i_frame
                cy0 = j_start+i_frame
                cy1 = j_end-i_frame
                count = integral_counts[cx1, cy1] - \
                        integral_counts[cx0, cy1] - \
                        integral_counts[cx1, cy0] + \
                        integral_counts[cx0, cy0]

                if threshold <= count <= max_threshold:
                    # Initialize to the constant term for each part and -inf to bkg
                    out_map_mv[i_start,j_start,0] = NINF 
                    for k in range(num_parts):
                        #out_map_mv[i_start,j_start,1+k] = constant_terms_mv[k]
                        out_map_mv[i_start,j_start,1+k] = 0.0# constant_terms_mv[k]

                    #for i in range(i_offset_x, part_x_dim, i_strides):
                        #for j in range(i_offset_y, part_y_dim, i_strides):

                    for index in range(num_indices):
                        for k in range(num_parts):
                            i = indices_mv[k,index,0]
                            j = indices_mv[k,index,1]
                            z = indices_mv[k,index,2]
                            #if X_mv[i_start+i,j_start+j,z]:
                            b = X_mv[i_start+i,j_start+j,z]
                            out_map_mv[i_start,j_start,k+1] += b * log_parts_mv[k,i,j,z] + (1 - b) * log_invparts_mv[k,i,j,z] 
                else:
                    out_map_mv[i_start,j_start,0] = 0.0
                    # Rest is already -inf
                
    return out_map


def code_parts__OLD(np.ndarray[ndim=3,dtype=UINT_t] X,
               np.ndarray[ndim=4,dtype=np.float64_t] log_parts,
               np.ndarray[ndim=4,dtype=np.float64_t] log_invparts,
               int threshold, outer_frame=0, strides=1, int max_threshold=10000):
    """
    At each location of `X`, find the log probabilities for each part and location. Outputs these part assignments in the same data dimensions as `X`. Neighborhoods of `X` with edge counts lower than `threshold` are regarded as background and assigned zero.

    Parameters
    ----------
    X : ndarray[ndim=3,dtype=np.uint8]
        The first two dimensions of the array specify locations. The last one specifies a binary edge type. The value ``X[s,t,e]`` is 1 iff there is an edge of type `e` detected at location `(s,t)`.
    log_parts : ndarray[ndim=4]
        We have a Bernoulli mixture model defined over patches of the input image. The `log_parts` is a logarithm applied to the array of edge probability maps for each part. Array of shape `(K, S, T, E)`, where `K` is the number of mixture component, `S` and `T` the shape of the data, and `E` the number of edges. The value of ``log_parts[k,s,t,e]`` is the log probability of observing an edge `e` at location `(s,t)`, conditioned on the mixture component being `k`.
    log_invparts : ndarray[ndim=4]
        Preprocessed inverse of `log_parts`, i.e. ``log(1-exp(log_parts))``.
    threshold : int
        The least number of edges in a patch to reject the null background hypothesis.
    outer_frame : int
        Remove a frame of this thickness when checking the threshold. If the parts are 5 x 5, and this is set to 1, then only the center 3 x 3 is used to count edges when compared to the threshold. 
    strides : int
        When checking a part, these are the strides in both axes. For instance, if we have 9 x 9 parts and strides is set to 3, then it will only check 3 x 3 locations.
    
    Returns
    -------
    out_map : ndarray[ndim=3] 
        Array of shape `(S, T, K+1)`. There are two cases, either the third dimension is `(0, -inf, -inf, ...)`, when there are insufficient edges in the neighborhood of a location. Otherwise, `out_map[s,t,i+1]` is the log likelihood of part `i` at location `(s,t)`. Additionally, `out_map[s,t,0]` is equal to `-inf`.
    """

    cdef unsigned int num_parts = log_parts.shape[0]
    cdef unsigned int part_x_dim = log_parts.shape[1]
    cdef unsigned int part_y_dim = log_parts.shape[2]
    cdef unsigned int part_z_dim = log_parts.shape[3]
    cdef unsigned int X_x_dim = X.shape[0]
    cdef unsigned int X_y_dim = X.shape[1]
    cdef unsigned int X_z_dim = X.shape[2]
    cdef unsigned int new_x_dim = X_x_dim - part_x_dim + 1
    cdef unsigned int new_y_dim = X_y_dim - part_y_dim + 1
    cdef unsigned int i_start,j_start,i_end,j_end,count,i,j,z,k, cx0, cx1, cy0, cy1 
    cdef unsigned int i_frame = <unsigned int>outer_frame
    cdef int i_strides = <int>strides 
    cdef int i_offset = i_strides / 2
    cdef DTYPE_t NINF = DTYPE(-np.inf)
    # we have num_parts + 1 because we are also including some regions as being
    # thresholded due to there not being enough edges
    
    cdef np.ndarray[dtype=DTYPE_t, ndim=3] out_map = np.ones((new_x_dim,
                                                              new_y_dim,
                                                              num_parts+1),dtype=DTYPE) * DTYPE(-np.inf)
    cdef UINT_t[:,:,:] X_mv = X

    cdef np.ndarray[dtype=DTYPE_t, ndim=4] part_logits = (log_parts - log_invparts).astype(DTYPE)

    cdef np.ndarray[dtype=DTYPE_t, ndim=1] constant_terms = np.apply_over_axes(np.sum, log_invparts[:,i_offset::i_strides,i_offset::i_strides].astype(DTYPE), [1, 2, 3]).ravel()
    cdef DTYPE_t[:,:,:,:] part_logits_mv = part_logits
    cdef DTYPE_t[:] constant_terms_mv = constant_terms

    cdef DTYPE_t[:,:,:] out_map_mv = out_map

    cdef np.ndarray[dtype=UINT32_t, ndim=2] _integral_counts = np.zeros((X_x_dim+1, X_y_dim+1), dtype=UINT32)
    cdef UINT32_t[:,:] integral_counts = _integral_counts

    # The first cell along the num_parts+1 axis contains a value that is either 0
    # if the area is deemed to have too few edges or min_val if there are sufficiently many
    # edges, min_val is just meant to be less than the value of the other cells
    # so when we pick the most likely part it won't be chosen

    with nogil:
        # Build integral image of edge counts
        # First, fill it with edge counts and accmulate across
        # one axis.
        for i in range(X_x_dim):
            for j in range(X_y_dim):
                count = 0
                for z in range(X_z_dim):
                    count += X_mv[i,j,z]
                integral_counts[1+i,1+j] = integral_counts[1+i,j] + count
        # Now accumulate the other axis
        for j in range(X_y_dim):
            for i in range(X_x_dim):
                integral_counts[1+i,1+j] += integral_counts[i,1+j]



        # Code parts
        for i_start in range(X_x_dim-part_x_dim+1):
            i_end = i_start + part_x_dim
            for j_start in range(X_y_dim-part_y_dim+1):
                j_end = j_start + part_y_dim
                #count = _count_edges(X_mv,i_start+i_frame,i_end-i_frame,j_start+i_frame,j_end-i_frame,X_z_dim)

                # Note, integral_counts is 1-based, to allow for a zero row/column at the zero:th index.
                cx0 = i_start+i_frame
                cx1 = i_end-i_frame
                cy0 = j_start+i_frame
                cy1 = j_end-i_frame
                count = integral_counts[cx1, cy1] - \
                        integral_counts[cx0, cy1] - \
                        integral_counts[cx1, cy0] + \
                        integral_counts[cx0, cy0]

                if threshold <= count <= max_threshold:
                    # Initialize to the constant term for each part and -inf to bkg
                    out_map_mv[i_start,j_start,0] = NINF 
                    for k in range(num_parts):
                        out_map_mv[i_start,j_start,1+k] = constant_terms_mv[k]

                    #for i in range(i_offset_x, part_x_dim, i_strides):
                        #for j in range(i_offset_y, part_y_dim, i_strides):
                    i = i_offset
                    while i < part_x_dim:
                        j = i_offset
                        while j < part_y_dim: 
                            for z in range(X_z_dim):
                                if X_mv[i_start+i,j_start+j,z] == 1:
                                    for k in range(num_parts):
                                        out_map_mv[i_start,j_start,k+1] += part_logits_mv[k,i,j,z]

                            j += i_strides
                        i += i_strides
                else:
                    out_map_mv[i_start,j_start,0] = 0.0
                    # Rest is already -inf
                
    return out_map

def code_parts_many(np.ndarray[ndim=4,dtype=UINT_t] X,
                    np.ndarray[ndim=4,dtype=DTYPE_t] log_parts,
                    np.ndarray[ndim=4,dtype=DTYPE_t] log_invparts,
                    int threshold, outer_frame=0):
    """
    At each location of `X`, find the log probabilities for each part and location. Outputs these part assignments in the same data dimensions as `X`. Neighborhoods of `X` with edge counts lower than `threshold` are regarded as background and assigned zero.

    Parameters
    ----------
    X : ndarray[ndim=3,dtype=np.uint8]
        The first two dimensions of the array specify locations. The last one specifies a binary edge type. The value ``X[s,t,e]`` is 1 iff there is an edge of type `e` detected at location `(s,t)`.
    log_parts : ndarray[ndim=4]
        We have a Bernoulli mixture model defined over patches of the input image. The `log_parts` is a logarithm applied to the array of edge probability maps for each part. Array of shape `(K, S, T, E)`, where `K` is the number of mixture component, `S` and `T` the shape of the data, and `E` the number of edges. The value of ``log_parts[k,s,t,e]`` is the log probability of observing an edge `e` at location `(s,t)`, conditioned on the mixture component being `k`.
    log_invparts : ndarray[ndim=4]
        Preprocessed inverse of `log_parts`, i.e. ``log(1-exp(log_parts))``.
    threshold : int
        The least number of edges in a patch to reject the null background hypothesis.
    outer_frame : int
        Remove a frame of this thickness when checking the threshold. If the parts are 5 x 5, and this is set to 1, then only the center 3 x 3 is used to count edges when compared to the threshold. 
    
    Returns
    -------
    out_map : ndarray[ndim=3] 
        Array of shape `(S, T, K+1)`. There are two cases, either the third dimension is `(0, -inf, -inf, ...)`, when there are insufficient edges in the neighborhood of a location. Otherwise, `out_map[s,t,i+1]` is the log likelihood of part `i` at location `(s,t)`. Additionally, `out_map[s,t,0]` is equal to `-inf`.
    """

    cdef unsigned int num_parts = log_parts.shape[0]
    cdef unsigned int part_x_dim = log_parts.shape[1]
    cdef unsigned int part_y_dim = log_parts.shape[2]
    cdef unsigned int part_z_dim = log_parts.shape[3]
    cdef unsigned int X_N = X.shape[0]
    cdef unsigned int X_x_dim = X.shape[1]
    cdef unsigned int X_y_dim = X.shape[2]
    cdef unsigned int X_z_dim = X.shape[3]
    cdef unsigned int new_x_dim = X_x_dim - part_x_dim + 1
    cdef unsigned int new_y_dim = X_y_dim - part_y_dim + 1
    cdef unsigned int i_start,j_start,i_end,j_end,count,i,j,z,k, cx0, cx1, cy0, cy1 
    cdef unsigned int i_frame = <unsigned int>outer_frame
    cdef DTYPE_t NINF = -np.inf
    # we have num_parts + 1 because we are also including some regions as being
    # thresholded due to there not being enough edges
    
    cdef np.ndarray[dtype=DTYPE_t, ndim=4] out_map = -np.inf * np.ones((X_N,
                                                                        new_x_dim,
                                                                        new_y_dim,
                                                                        num_parts+1),dtype=DTYPE)
    cdef UINT_t[:,:,:,:] X_mv = X
    cdef DTYPE_t[:,:,:,:] log_parts_mv = log_parts
    cdef DTYPE_t[:,:,:,:] log_invparts_mv = log_invparts
    cdef DTYPE_t[:,:,:,:] out_map_mv = out_map

    cdef np.ndarray[dtype=UINT32_t, ndim=2] _integral_counts = np.zeros((X_x_dim+1, X_y_dim+1), dtype=UINT32)
    cdef UINT32_t[:,:] integral_counts = _integral_counts

    #cdef UINT_t[:,:]
    # The first cell along the num_parts+1 axis contains a value that is either 0
    # if the area is deemed to have too few edges or min_val if there are sufficiently many
    # edges, min_val is just meant to be less than the value of the other cells
    # so when we pick the most likely part it won't be chosen

    with nogil:
        for n in range(X_N):
            # Build integral image of edge counts
            # First, fill it with edge counts and accmulate across
            # one axis.
            for i in range(X_x_dim):
                for j in range(X_y_dim):
                    count = 0
                    for z in range(X_z_dim):
                        count += X_mv[n,i,j,z]
                    integral_counts[1+i,1+j] = integral_counts[1+i,j] + count
            # Now accumulate the other axis
            for j in range(X_y_dim):
                for i in range(X_x_dim):
                    integral_counts[1+i,1+j] += integral_counts[i,1+j]



            # Code parts
            for i_start in range(X_x_dim-part_x_dim+1):
                i_end = i_start + part_x_dim
                for j_start in range(X_y_dim-part_y_dim+1):
                    j_end = j_start + part_y_dim
                    #count = _count_edges(X_mv,i_start+i_frame,i_end-i_frame,j_start+i_frame,j_end-i_frame,X_z_dim)

                    # Note, integral_counts is 1-based, to allow for a zero row/column at the zero:th index.
                    cx0 = i_start+i_frame
                    cx1 = i_end-i_frame
                    cy0 = j_start+i_frame
                    cy1 = j_end-i_frame
                    count = integral_counts[cx1, cy1] - \
                            integral_counts[cx0, cy1] - \
                            integral_counts[cx1, cy0] + \
                            integral_counts[cx0, cy0]

                    if count >= threshold:
                        out_map_mv[n,i_start,j_start] = 0.0
                        out_map_mv[n,i_start,j_start,0] = NINF 
                        for i in range(part_x_dim):
                            for j in range(part_y_dim):
                                for z in range(X_z_dim):
                                    if X_mv[n,i_start+i,j_start+j,z]:
                                        for k in range(num_parts):
                                            out_map_mv[n,i_start,j_start,k+1] += log_parts_mv[k,i,j,z]
                                    else:
                                        for k in range(num_parts):
                                            out_map_mv[n,i_start,j_start,k+1] += log_invparts_mv[k,i,j,z]
                    else:
                        out_map_mv[n,i_start,j_start,0] = 0.0
                    
    return out_map

def code_parts_mask(np.ndarray[ndim=3,dtype=UINT_t] X,
               np.ndarray[ndim=4,dtype=DTYPE_t] log_parts,
               np.ndarray[ndim=4,dtype=DTYPE_t] log_invparts,
                    int threshold, 
                    np.ndarray[ndim=2,dtype=UINT_t] M):
    """
    At each location of `X`, find the log probabilities for each part and location. Outputs these part assignments in the same data dimensions as `X`. Neighborhoods of `X` with edge counts lower than `threshold` are regarded as background and assigned zero.

    Parameters
    ----------
    X : ndarray[ndim=3,dtype=np.uint8]
        The first two dimensions of the array specify locations. The last one specifies a binary edge type. The value ``X[s,t,e]`` is 1 iff there is an edge of type `e` detected at location `(s,t)`.
    log_parts : ndarray[ndim=4]
        We have a Bernoulli mixture model defined over patches of the input image. The `log_parts` is a logarithm applied to the array of edge probability maps for each part. Array of shape `(K, S, T, E)`, where `K` is the number of mixture component, `S` and `T` the shape of the data, and `E` the number of edges. The value of ``log_parts[k,s,t,e]`` is the log probability of observing an edge `e` at location `(s,t)`, conditioned on the mixture component being `k`.
    log_invparts : ndarray[ndim=4]
        Preprocessed inverse of `log_parts`, i.e. ``log(1-exp(log_parts))``.
    threshold : int
        The least number of edges in a patch to reject the null background hypothesis.
    M : ndarray[ndim=2]
        Mask over the patches that determines which X-Y locations will be used for computing whether the number of edges reaches a certain threshold. Should be a binary array
    
    Returns
    -------
    out_map : ndarray[ndim=3] 
        Array of shape `(S, T, K+1)`. There are two cases, either the third dimension is `(0, -inf, -inf, ...)`, when there are insufficient edges in the neighborhood of a location. Otherwise, `out_map[s,t,i+1]` is the log likelihood of part `i` at location `(s,t)`. Additionally, `out_map[s,t,0]` is equal to `-inf`.
    """

    cdef unsigned int num_parts = log_parts.shape[0]
    cdef unsigned int part_x_dim = log_parts.shape[1]
    cdef unsigned int part_y_dim = log_parts.shape[2]
    cdef unsigned int part_z_dim = log_parts.shape[3]
    cdef unsigned int X_x_dim = X.shape[0]
    cdef unsigned int X_y_dim = X.shape[1]
    cdef unsigned int X_z_dim = X.shape[2]
    cdef unsigned int new_x_dim = X_x_dim - part_x_dim + 1
    cdef unsigned int new_y_dim = X_y_dim - part_y_dim + 1
    cdef unsigned int i_start,j_start,i_end,j_end,count,i,j,z,k
    cdef DTYPE_t NINF = -np.inf
    # we have num_parts + 1 because we are also including some regions as being
    # thresholded due to there not being enough edges
    

    cdef np.ndarray[dtype=DTYPE_t, ndim=3] out_map = -np.inf * np.ones((new_x_dim,
                                                                        new_y_dim,
                                                                        num_parts+1),dtype=DTYPE)
    cdef UINT_t[:,:,:] X_mv = X
    cdef DTYPE_t[:,:,:,:] log_parts_mv = log_parts
    cdef DTYPE_t[:,:,:,:] log_invparts_mv = log_invparts
    cdef DTYPE_t[:,:,:] out_map_mv = out_map
    # The first cell along the num_parts+1 axis contains a value that is either 0
    # if the area is deemed to have too few edges or min_val if there are sufficiently many
    # edges, min_val is just meant to be less than the value of the other cells
    # so when we pick the most likely part it won't be chosen

    for i_start in range(X_x_dim-part_x_dim+1):
        i_end = i_start + part_x_dim
        for j_start in range(X_y_dim-part_y_dim+1):
            j_end = j_start + part_y_dim
            count = _count_edges_mask(X, M,
                                            i_start,i_end,j_start,j_end,X_z_dim)

            if count >= threshold:
                out_map_mv[i_start,j_start] = 0.0
                out_map_mv[i_start,j_start,0] = NINF 
                for i in range(part_x_dim):
                    for j in range(part_y_dim):
                        for z in range(X_z_dim):
                            if X_mv[i_start+i,j_start+j,z]:
                                for k in range(num_parts):
                                    out_map_mv[i_start,j_start,k+1] += log_parts_mv[k,i,j,z]
                            else:
                                for k in range(num_parts):
                                    out_map_mv[i_start,j_start,k+1] += log_invparts_mv[k,i,j,z]
            else:
                out_map_mv[i_start,j_start,0] = 0.0
                
    return out_map

# TODO: Experimentally adding --Gustav
def code_parts_support_mask(np.ndarray[ndim=3,dtype=UINT_t] X,
               np.ndarray[ndim=4,dtype=DTYPE_t] log_parts,
               np.ndarray[ndim=4,dtype=DTYPE_t] log_invparts,
                    int threshold,
                    np.ndarray[ndim=2,dtype=UINT_t] M,
                    outer_frame = 0):
    """
    At each location of `X`, find the log probabilities for each part and location. Outputs these part assignments in the same data dimensions as `X`. Neighborhoods of `X` with edge counts lower than `threshold` are regarded as background and assigned zero.

    Parameters
    ----------
    X : ndarray[ndim=3,dtype=np.uint8]
        The first two dimensions of the array specify locations. The last one specifies a binary edge type. The value ``X[s,t,e]`` is 1 iff there is an edge of type `e` detected at location `(s,t)`.
    log_parts : ndarray[ndim=4]
        We have a Bernoulli mixture model defined over patches of the input image. The `log_parts` is a logarithm applied to the array of edge probability maps for each part. Array of shape `(K, S, T, E)`, where `K` is the number of mixture component, `S` and `T` the shape of the data, and `E` the number of edges. The value of ``log_parts[k,s,t,e]`` is the log probability of observing an edge `e` at location `(s,t)`, conditioned on the mixture component being `k`.
    log_invparts : ndarray[ndim=4]
        Preprocessed inverse of `log_parts`, i.e. ``log(1-exp(log_parts))``.
    threshold : int
        The least number of edges in a patch to reject the null background hypothesis.
    M : ndarray[ndim=2]
        Mask over the patches that determines which X-Y locations will be used for computing whether the number of edges reaches a certain threshold. Should be a binary array
    
    Returns
    -------
    out_map : ndarray[ndim=3] 
        Array of shape `(S, T, K+1)`. There are two cases, either the third dimension is `(0, -inf, -inf, ...)`, when there are insufficient edges in the neighborhood of a location. Otherwise, `out_map[s,t,i+1]` is the log likelihood of part `i` at location `(s,t)`. Additionally, `out_map[s,t,0]` is equal to `-inf`.
    """
    cdef unsigned int num_parts = log_parts.shape[0]
    cdef unsigned int part_x_dim = log_parts.shape[1]
    cdef unsigned int part_y_dim = log_parts.shape[2]
    cdef unsigned int part_z_dim = log_parts.shape[3]
    cdef unsigned int X_x_dim = X.shape[0]
    cdef unsigned int X_y_dim = X.shape[1]
    cdef unsigned int X_z_dim = X.shape[2]
    cdef unsigned int new_x_dim = X_x_dim - part_x_dim + 1
    cdef unsigned int new_y_dim = X_y_dim - part_y_dim + 1
    cdef unsigned int i_start,j_start,i_end,j_end,count,i,j,z,k, cx0, cx1, cy0, cy1 
    cdef unsigned int i_frame = <unsigned int>outer_frame
    cdef DTYPE_t NINF = -np.inf
    # we have num_parts + 1 because we are also including some regions as being
    # thresholded due to there not being enough edges
    

    cdef np.ndarray[dtype=DTYPE_t, ndim=3] out_map = -np.inf * np.ones((new_x_dim,
                                                                        new_y_dim,
                                                                        num_parts+1),dtype=DTYPE)
    cdef UINT_t[:,:,:] X_mv = X
    cdef DTYPE_t[:,:,:,:] log_parts_mv = log_parts
    cdef DTYPE_t[:,:,:,:] log_invparts_mv = log_invparts
    cdef UINT_t[:,:] M_mv = M
    cdef DTYPE_t[:,:,:] out_map_mv = out_map

    cdef np.ndarray[dtype=UINT32_t, ndim=2] _integral_counts = np.zeros((X_x_dim+1, X_y_dim+1), dtype=UINT32)
    cdef UINT32_t[:,:] integral_counts = _integral_counts

    #cdef UINT_t[:,:]
    # The first cell along the num_parts+1 axis contains a value that is either 0
    # if the area is deemed to have too few edges or min_val if there are sufficiently many
    # edges, min_val is just meant to be less than the value of the other cells
    # so when we pick the most likely part it won't be chosen

    with nogil:
        # Build integral image of edge counts
        # First, fill it with edge counts and accmulate across
        # one axis.
        for i in range(X_x_dim):
            for j in range(X_y_dim):
                count = 0
                for z in range(X_z_dim):
                    count += X_mv[i,j,z]
                integral_counts[1+i,1+j] = integral_counts[1+i,j] + count
        # Now accumulate the other axis
        for j in range(X_y_dim):
            for i in range(X_x_dim):
                integral_counts[1+i,1+j] += integral_counts[i,1+j]



        # Code parts
        for i_start in range(X_x_dim-part_x_dim+1):
            i_end = i_start + part_x_dim
            for j_start in range(X_y_dim-part_y_dim+1):
                j_end = j_start + part_y_dim
                #count = _count_edges(X_mv,i_start+i_frame,i_end-i_frame,j_start+i_frame,j_end-i_frame,X_z_dim)

                # Note, integral_counts is 1-based, to allow for a zero row/column at the zero:th index.
                cx0 = i_start+i_frame
                cx1 = 1+i_end-i_frame
                cy0 = j_start+i_frame
                cy1 = 1+j_end-i_frame
                count = integral_counts[cx1, cy1] - \
                        integral_counts[cx0, cy1] - \
                        integral_counts[cx1, cy0] + \
                        integral_counts[cx0, cy0]
            
                if count >= threshold:
                    out_map_mv[i_start,j_start] = 0.0
                    out_map_mv[i_start,j_start,0] = NINF 
                    for i in range(part_x_dim):
                        for j in range(part_y_dim):
                            if M_mv[i_start+i,j_start+j]:
                                for z in range(X_z_dim):
                                    # Only if it's inside the support
                                    if X_mv[i_start+i,j_start+j,z]:
                                        for k in range(num_parts):
                                            out_map_mv[i_start,j_start,k+1] += log_parts_mv[k,i,j,z]
                                    else:
                                        for k in range(num_parts):
                                            out_map_mv[i_start,j_start,k+1] += log_invparts_mv[k,i,j,z]
                else:
                    out_map_mv[i_start,j_start,0] = 0.0
                
    return out_map

def convert_part_to_feature_vector(np.ndarray[dtype=UINT32_t,ndim=2] one_indexed_parts, int num_parts):
    cdef np.uint16_t X_dim_0 = one_indexed_parts.shape[0]
    cdef np.uint16_t X_dim_1 = one_indexed_parts.shape[1]
    cdef int i, j, f, v
    cdef np.ndarray[dtype=UINT_t, ndim=3] feats = np.zeros((X_dim_0, X_dim_1, num_parts), dtype=UINT)
    cdef UINT_t[:,:,:] feats_mv = feats

    cdef UINT32_t[:,:] parts_mv = one_indexed_parts

    with nogil:
        for i in range(X_dim_0):
            for j in range(X_dim_1):
                v = parts_mv[i,j] 
                if v != 0:
                    feats_mv[i,j,v-1] = 1

    return feats

def convert_partprobs_to_feature_vector(np.ndarray[dtype=DTYPE_t,ndim=3] partprobs, tau=0.0):
    cdef np.uint16_t X_dim_0 = partprobs.shape[0]
    cdef np.uint16_t X_dim_1 = partprobs.shape[1]
    cdef int num_parts = partprobs.shape[2] - 1
    cdef int i, j, f, m
    cdef DTYPE_t d = 0.0, mx = 0.0
    cdef np.ndarray[dtype=UINT_t, ndim=3] feats = np.zeros((X_dim_0, X_dim_1, num_parts), dtype=UINT)
    cdef UINT_t[:,:,:] feats_mv = feats
    cdef DTYPE_t NINF = DTYPE(-np.inf)
    cdef DTYPE_t ttau = DTYPE(tau)

    cdef DTYPE_t[:,:,:] partprobs_mv = partprobs 

    with nogil:
        for i in range(X_dim_0):
            for j in range(X_dim_1):
                mx = NINF
                m = 0
                for f in range(1, num_parts+1):
                    if partprobs_mv[i,j,f] > mx: 
                        m = f
                        mx = partprobs_mv[i,j,f]
                if m != 0:
                    d = partprobs_mv[i,j,m] - ttau
                    for f in range(1,num_parts+1):
                        if partprobs_mv[i,j,f] >= d:
                            feats_mv[i,j,f-1] = 1 
    
    return feats

def code_parts_as_features(np.ndarray[ndim=3,dtype=UINT_t] X,
                           np.ndarray[ndim=3,dtype=UINT_t] X_unspread,
                           np.ndarray[ndim=4,dtype=np.float64_t] log_parts,
                           np.ndarray[ndim=4,dtype=np.float64_t] log_invparts,
                           int threshold, outer_frame=0, strides=1, tau=0.0, int max_threshold=10000):

    partprobs = code_parts(X, X_unspread, log_parts, log_invparts, threshold, outer_frame=outer_frame, strides=strides, max_threshold=max_threshold)
    if tau == 0.0:
        # This is a bit faster than the one for tau > 0.0 (even though the other one works too)
        return convert_part_to_feature_vector(partprobs.argmax(axis=-1).astype(UINT32), partprobs.shape[2]-1)
    else:
        return convert_partprobs_to_feature_vector(partprobs, tau)

def code_parts_as_features_INDICES(np.ndarray[ndim=3,dtype=UINT_t] X,
                           np.ndarray[ndim=3,dtype=UINT_t] X_unspread,
                           np.ndarray[ndim=4,dtype=np.float64_t] log_parts,
                           np.ndarray[ndim=4,dtype=np.float64_t] log_invparts,
                           np.ndarray[ndim=3,dtype=UINT32_t] indices,
                           int threshold, outer_frame=0, strides=1, tau=0.0, int max_threshold=10000):

    partprobs = code_parts_INDICES(X, X_unspread, log_parts, log_invparts, indices, threshold, outer_frame=outer_frame, strides=strides, max_threshold=max_threshold)
    if tau == 0.0:
        # This is a bit faster than the one for tau > 0.0 (even though the other one works too)
        return convert_part_to_feature_vector(partprobs.argmax(axis=-1).astype(UINT32), partprobs.shape[2]-1)
    else:
        return convert_partprobs_to_feature_vector(partprobs, tau)

def code_parts_as_features__OLD(np.ndarray[ndim=3,dtype=UINT_t] X,
                           np.ndarray[ndim=4,dtype=np.float64_t] log_parts,
                           np.ndarray[ndim=4,dtype=np.float64_t] log_invparts,
                           int threshold, outer_frame=0, strides=1, tau=0.0, int max_threshold=10000):

    partprobs = code_parts(X, log_parts, log_invparts, threshold, outer_frame=outer_frame, strides=strides, max_threshold=max_threshold)
    if tau == 0.0:
        # This is a bit faster than the one for tau > 0.0 (even though than one works too)
        return convert_part_to_feature_vector(partprobs.argmax(axis=-1).astype(UINT32), partprobs.shape[2]-1)
    else:
        return convert_partprobs_to_feature_vector(partprobs, tau)

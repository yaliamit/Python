#!python
# cython: embedsignature=True
# cython: cdivision=True
# cython: boundscheck=False
import cython
import numpy as np
import colorsys
cimport numpy as np
#cimport colorsys
#from cython.parallel import prange
DTYPE = np.float64
BTYPE = np.uint8
ctypedef np.float64_t DTYPE_t
from libc.math cimport fabs

cdef inline int int_max(int a, int b) nogil: return a if a >= b else b
cdef inline int int_min(int a, int b) nogil: return a if a <= b else b
cdef inline int int_abs(int a) nogil: return a if a >= 0 else -a

cdef inline void _checkedge(DTYPE_t[:,:,:] images, np.uint8_t[:,:,:,:] ret, int ii, int vi, int z0, int z1, int v0, int v1, int w0, int w1, int k, double minimum_contrast, int displace) nogil:
    cdef int y0 = z0 + v0
    cdef int y1 = z1 + v1
    cdef DTYPE_t Iy = images[ii, y0, y1] 
    cdef DTYPE_t Iz = images[ii, z0, z1] 
    
    cdef DTYPE_t d = fabs(Iy - Iz)
    cdef int num_edges = <int>(d > fabs(images[ii, z0+w0, z1+w1] - Iz)) + \
                         <int>(d > fabs(images[ii, y0+w0, y1+w1] - Iy)) + \
                         <int>(d > fabs(images[ii, z0-w0, z1-w1] - Iz)) + \
                         <int>(d > fabs(images[ii, y0-w0, y1-w1] - Iy)) + \
                         <int>(d > fabs(images[ii, z0-v0, z1-v1] - Iz)) + \
                         <int>(d > fabs(images[ii, y0+v0, y1+v1] - Iy))

    if num_edges >= k and d > minimum_contrast: 
        ret[ii, vi + displace*<int>(Iy > Iz), z0, z1] = 1 

def array_bedges(np.ndarray[DTYPE_t, ndim=3] images, int k, minimum_contrast, contrast_insensitive):
    assert(images.dtype == DTYPE)
    cdef int N = images.shape[0]
    cdef int rows = images.shape[1]
    cdef int cols = images.shape[2] 
    cdef DTYPE_t[:,:,:] images_mv = images
    cdef Py_ssize_t i
    cdef int z0
    cdef int z1
    cdef int int_k = <int>k
    cdef double double_minimum_contrast = <double>minimum_contrast
    cdef int displace = 0
    cdef int binary_features = 8

    if contrast_insensitive:
        displace = 0
        binary_features = 4
    else:
        displace = 4 
        binary_features = 8

    cdef np.ndarray[np.uint8_t, ndim=4] ret = np.zeros((N, binary_features, rows, cols), dtype=np.uint8)
    cdef np.uint8_t[:,:,:,:] ret_mv = ret
    
    #for i in prange(N, nogil=True):
    with nogil:
        for i in range(N):
            for z0 in range(2, rows-2):
                for z1 in range(2, cols-2):
                    _checkedge(images_mv, ret_mv, i, 0, z0, z1, 1, 0, 0, -1, int_k, double_minimum_contrast, displace)
                    _checkedge(images_mv, ret_mv, i, 1, z0, z1, 1, 1, 1, -1, int_k, double_minimum_contrast, displace)
                    _checkedge(images_mv, ret_mv, i, 2, z0, z1, 0, 1, 1, 0, int_k, double_minimum_contrast, displace)
                    _checkedge(images_mv, ret_mv, i, 3, z0, z1, -1, 1, 1, 1, int_k, double_minimum_contrast, displace)

    return ret

# BYTE VERSION

cdef inline void _byte_checkedge(np.uint8_t[:,:,:] images, np.uint8_t[:,:,:,:] ret, int ii, int vi, int z0, int z1, int v0, int v1, int w0, int w1, int k, int minimum_contrast, int displace) nogil:
    cdef int y0 = z0 + v0
    cdef int y1 = z1 + v1
    cdef np.uint8_t Iy = images[ii, y0, y1] 
    cdef np.uint8_t Iz = images[ii, z0, z1] 
    
    cdef np.uint8_t d = int_abs(Iy - Iz)
    cdef int num_edges = <int>(d > int_abs(images[ii, z0+w0, z1+w1] - Iz)) + \
                         <int>(d > int_abs(images[ii, y0+w0, y1+w1] - Iy)) + \
                         <int>(d > int_abs(images[ii, z0-w0, z1-w1] - Iz)) + \
                         <int>(d > int_abs(images[ii, y0-w0, y1-w1] - Iy)) + \
                         <int>(d > int_abs(images[ii, z0-v0, z1-v1] - Iz)) + \
                         <int>(d > int_abs(images[ii, y0+v0, y1+v1] - Iy))

    if num_edges >= k and d > minimum_contrast: 
        ret[ii, vi + displace*<int>(Iy > Iz), z0, z1] = 1 


# def change_saturation_c(np.ndarray[np.floatX_t,ndim=3] xx,float fac):
#
#     cdef int rows = xx.shape[0]
#     cdef int cols = xx.shape[1]
#     cdef np.ndarray[np.floatX_t, ndim=3] yy = np.zeros((rows, cols, 3), dtype=np.floatX)
#     cdef np.ndarray[np.floatX_t, ndim=1] uu = np.zeros(3, dtype=np.floatX)
#
#     for i in range(rows):
#         for j in range(cols):
#             uu=np.array(colorsys.rgb_to_hsv(xx[i,j,0],xx[i,j,1],xx[i,j,2]),dtype=np.floatX)
#             uu[1]=np.minimum(uu[1]*fac,1)
#             uu=np.array(colorsys.hsv_to_rgb(uu[0],uu[1],uu[2]),dtype=np.floatX)
#             for k in range(3):
#                 yy[i,j,k]=uu[k]
#
#
#     return(yy)


def change_saturation_c(np.ndarray[np.floatX_t,ndim=4] xx,np.ndarray[np.floatX_t, ndim=1] fac):

    cdef int N = xx.shape[0]
    cdef int rows = xx.shape[1]
    cdef int cols = xx.shape[2]
    cdef np.ndarray[np.floatX_t, ndim=4] yy = np.zeros((N,rows, cols, 3), dtype=np.floatX)
    cdef np.ndarray[np.floatX_t, ndim=1] uu = np.zeros(3, dtype=np.floatX)
    for n in range(N):
        for i in range(rows):
            for j in range(cols):
                uu=np.array(colorsys.rgb_to_hsv(xx[n,i,j,0],xx[n,i,j,1],xx[n,i,j,2]),dtype=np.floatX)
                uu[1]=np.minimum(uu[1]*fac[n],1)
                uu=np.array(colorsys.hsv_to_rgb(uu[0],uu[1],uu[2]),dtype=np.floatX)
                for k in range(3):
                    yy[n,i,j,k]=uu[k]


    return(yy)


def array_byte_bedges(np.ndarray[np.uint8_t, ndim=3] images, int k, int minimum_contrast, contrast_insensitive):
    assert(images.dtype == np.uint8)
    cdef int N = images.shape[0]
    cdef int rows = images.shape[1]
    cdef int cols = images.shape[2] 
    cdef np.uint8_t[:,:,:] images_mv = images
    cdef Py_ssize_t i
    cdef int z0
    cdef int z1
    cdef int displace = 0
    cdef int binary_features = 8

    if contrast_insensitive:
        displace = 0
        binary_features = 4
    else:
        displace = 4 
        binary_features = 8

    cdef np.ndarray[np.uint8_t, ndim=4] ret = np.zeros((N, binary_features, rows, cols), dtype=np.uint8)
    cdef np.uint8_t[:,:,:,:] ret_mv = ret
    
    #for i in prange(N, nogil=True):
    with nogil:
        for i in range(N):
            for z0 in range(2, rows-2):
                for z1 in range(2, cols-2):
                    _byte_checkedge(images_mv, ret_mv, i, 0, z0, z1, 1, 0, 0, -1, k, minimum_contrast, displace)
                    _byte_checkedge(images_mv, ret_mv, i, 1, z0, z1, 1, 1, 1, -1, k, minimum_contrast, displace)
                    _byte_checkedge(images_mv, ret_mv, i, 2, z0, z1, 0, 1, 1, 0, k, minimum_contrast, displace)
                    _byte_checkedge(images_mv, ret_mv, i, 3, z0, z1, -1, 1, 1, 1, k, minimum_contrast, displace)

    return ret.view(np.bool_)



########################################################################
cdef inline DTYPE_t _checkedge2(DTYPE_t[:,:,:] images, int ii, int vi, int z0, int z1, int v0, int v1, int w0, int w1, int k, double minimum_contrast, int displace, int *polarity) nogil:
    cdef int y0 = z0 + v0
    cdef int y1 = z1 + v1
    cdef DTYPE_t m
    cdef DTYPE_t Iy = images[ii, y0, y1] 
    cdef DTYPE_t Iz = images[ii, z0, z1] 
    
    cdef DTYPE_t d = fabs(Iy - Iz)
    cdef int num_edges = <int>(d > fabs(images[ii, z0+w0, z1+w1] - Iz)) + \
                         <int>(d > fabs(images[ii, y0+w0, y1+w1] - Iy)) + \
                         <int>(d > fabs(images[ii, z0-w0, z1-w1] - Iz)) + \
                         <int>(d > fabs(images[ii, y0-w0, y1-w1] - Iy)) + \
                         <int>(d > fabs(images[ii, z0-v0, z1-v1] - Iz)) + \
                         <int>(d > fabs(images[ii, y0+v0, y1+v1] - Iy))

    #if num_edges >= k and d > minimum_contrast: 
        #ret[ii, vi + displace*<int>(Iy > Iz), z0, z1] = 1 
    polarity[0] = <int>(Iy > Iz) 
    if num_edges >= k and d > minimum_contrast:
        return d
    else:
        return 0.0

def array_bedges2(np.ndarray[DTYPE_t, ndim=3] images, k, minimum_contrast, contrast_insensitive, max_edges):
    assert(images.dtype == DTYPE)
    cdef int N = images.shape[0]
    cdef int rows = images.shape[1]
    cdef int cols = images.shape[2] 
    cdef DTYPE_t[:,:,:] images_mv = images
    cdef Py_ssize_t i
    cdef int j
    cdef int z0
    cdef int z1
    cdef int int_k = <int>k
    cdef double double_minimum_contrast = <double>minimum_contrast
    cdef int displace = 0
    cdef int binary_features = 8
    cdef int int_max_edges = <int>max_edges

    if contrast_insensitive:
        displace = 0
        binary_features = 4
    else:
        displace = 4 
        binary_features = 8

    cdef np.ndarray[np.uint8_t, ndim=4] ret = np.zeros((N, binary_features, rows, cols), dtype=np.uint8)
    cdef np.uint8_t[:,:,:,:] ret_mv = ret

    cdef int pol[4]
    cdef int max_con_index
    cdef DTYPE_t max_con
    cdef DTYPE_t con[4] 
    
    #for i in prange(N, nogil=True):
    with nogil:
        for i in range(N):
            for z0 in range(2, rows-2):
                for z1 in range(2, cols-2):
                    con[0] = _checkedge2(images_mv, i, 0, z0, z1, 1, 0, 0, -1, int_k, double_minimum_contrast, displace, &pol[0])
                    con[1] = _checkedge2(images_mv, i, 1, z0, z1, 1, 1, 1, -1, int_k, double_minimum_contrast, displace, &pol[1])
                    con[2] = _checkedge2(images_mv, i, 2, z0, z1, 0, 1, 1,  0, int_k, double_minimum_contrast, displace, &pol[2])
                    con[3] = _checkedge2(images_mv, i, 3, z0, z1, -1, 1, 1, 1, int_k, double_minimum_contrast, displace, &pol[3])

                    for j in range(int_max_edges):
                        max_con = 0.0
                        max_con_index = -1
                        for l in range(4):
                            if con[l] > max_con:
                                max_con_index = l
                        if max_con_index != -1:
                            ret[i, max_con_index + displace*pol[max_con_index], z0, z1] = 1
                            con[max_con_index] = 0.0 
                            #ret[ii, vi + displace*<int>(Iy > Iz), z0, z1] = 1 
                            
    return ret

def array_intensities(np.ndarray[DTYPE_t, ndim=3] images, k, minimum_contrast, contrast_insensitive):
    assert(images.dtype == DTYPE)
    cdef int N = images.shape[0]
    cdef int rows = images.shape[1]
    cdef int cols = images.shape[2] 
    cdef DTYPE_t[:,:,:] images_mv = images
    cdef Py_ssize_t i
    cdef int j
    cdef int z0
    cdef int z1
    cdef int int_k = <int>k
    cdef double double_minimum_contrast = <double>minimum_contrast
    cdef int displace = 0
    cdef int binary_features = 8

    if contrast_insensitive:
        displace = 0
        binary_features = 4
    else:
        displace = 4 
        binary_features = 8

    cdef np.ndarray[np.uint8_t, ndim=4] ret = np.zeros((N, binary_features, rows, cols), dtype=np.uint8)
    cdef np.uint8_t[:,:,:,:] ret_mv = ret

    cdef np.ndarray[DTYPE_t, ndim=4] intensities = np.zeros((N, binary_features, rows, cols), dtype=DTYPE)
    cdef DTYPE_t[:,:,:,:] intensities_mv = intensities

    cdef int pol[4]
    cdef int max_con_index
    cdef DTYPE_t max_con
    cdef DTYPE_t con[4] 
    
    #for i in prange(N, nogil=True):
    with nogil:
        for i in range(N):
            for z0 in range(2, rows-2):
                for z1 in range(2, cols-2):
                    con[0] = _checkedge2(images_mv, i, 0, z0, z1, 1, 0, 0, -1, int_k, double_minimum_contrast, displace, &pol[0])
                    con[1] = _checkedge2(images_mv, i, 1, z0, z1, 1, 1, 1, -1, int_k, double_minimum_contrast, displace, &pol[1])
                    con[2] = _checkedge2(images_mv, i, 2, z0, z1, 0, 1, 1,  0, int_k, double_minimum_contrast, displace, &pol[2])
                    con[3] = _checkedge2(images_mv, i, 3, z0, z1, -1, 1, 1, 1, int_k, double_minimum_contrast, displace, &pol[3])

                    for j in range(4):
                        intensities[i, j + displace*pol[j], z0, z1] = con[j]
                            
    return intensities

def array_bspread(np.ndarray[np.uint8_t, ndim=4] X, spread='box', radius=1):
    cdef:
        np.uint8_t[:,:,:,:] X_mv = X
        int N = X.shape[0]
        int E = X.shape[1]
        int dim0 = X.shape[2]
        int dim1 = X.shape[3]
        int n, e, i, j, x, y, loop, rounds, offset
        int lo_spread_0_idx,hi_spread_0_idx,lo_spread_1_idx,hi_spread_1_idx
        int i_radius = <int>radius
        np.uint8_t v

        np.ndarray[np.uint8_t, ndim=4] ret = np.zeros_like(X)
        np.uint8_t[:,:,:,:] ret_mv = ret

        #np.ndarray[np.uint8_t, ndim=4] intermediate_ret = np.zeros_like(X)
        #np.uint8_t[:,:,:,:] intermediate_ret_mv = intermediate_ret

    if spread == 'box' or spread is True:
        # Since box spreading is separable, we will first spread the first dimension
        # and then the second
        with nogil:
            if 0:
                for n in range(N):
                    for e in range(E):
                        for i in range(dim0):
                            for j in range(dim1):   
                                v = X_mv[n,e,i,j]
                                if v == 1:
                                    for x in range(int_max(0, i-i_radius), int_min(dim0, i+i_radius+1)):
                                        intermediate_ret_mv[n,e,x,j] = 1

                for n in range(N):
                    for e in range(E):
                        for i in range(dim0):
                            for j in range(dim1):   
                                v = intermediate_ret_mv[n,e,i,j]
                                if v == 1:
                                    for y in range(int_max(0, j-i_radius), int_min(dim1, j+i_radius+1)):
                                        ret_mv[n,e,i,y] = 1

        with nogil:
            for n in range(N):
                for e in range(E):
                    for i in range(dim0):
                        lo_spread_0_idx = max(i-i_radius,0)
                        hi_spread_0_idx = min(i+i_radius+1,dim0)
                        for j in range(dim1):   
                            lo_spread_1_idx = max(j-i_radius,0)
                            hi_spread_1_idx = min(j+i_radius+1,dim1)
                            v = X_mv[n,e,i,j]
                            if v == 1:
                                for x in range(lo_spread_0_idx, hi_spread_0_idx):
                                    for y in range(lo_spread_1_idx, hi_spread_1_idx):
                                        ret_mv[n,e,x,y] = 1 
    elif spread == 'orthogonal':
        with nogil:
            rounds = E/4
            for n in range(N):
                for loop in range(rounds):
                    offset = 4 * loop

                    for i in range(dim0):
                        for j in range(dim1):   
                            # N/S
                            v = X_mv[n,offset,i,j]
                            if v:
                                for x in range(int_max(0, i-i_radius), int_min(dim0, i+i_radius+1)):
                                    ret_mv[n,offset,x,j] = 1 

                            # E/W
                            v = X_mv[n,offset+2,i,j]
                            if v:
                                for y in range(int_max(0, j-i_radius), int_min(dim1, j+i_radius+1)):
                                    ret_mv[n,offset+2,i,y] = 1 

                            # Diagonal (along matrix diagonal)
                            v = X_mv[n,offset+1,i,j]
                            if v:
                                for x in range(-int_min(int_min(i, j), i_radius), int_min(int_min(dim0-i, dim1-j), i_radius+1)):
                                    ret_mv[n,offset+1,i+x,j+x] = 1

                            # Diagonal (along matrix anti-diagonal)
                            v = X_mv[n,offset+3,i,j]
                            if v:
                                for x in range(-int_min(int_min(i, dim1-j-1), i_radius), int_min(int_min(dim0-i, j+1), i_radius+1)):
                                    ret_mv[n,offset+3,i+x,j-x] = 1
    else:
        raise ValueError("Unrecognized spreading method: {0}".format(spread))

    return ret

def array_bspread_new(np.ndarray[np.uint8_t, ndim=3] X, spread='box', radius=1):
    cdef:
        np.uint8_t[:,:,:] X_mv = X
        int E = X.shape[2]
        int dim0 = X.shape[0]
        int dim1 = X.shape[1]
        int n, e, i, j, x, y, loop, rounds, offset
        int lo_spread_0_idx,hi_spread_0_idx,lo_spread_1_idx,hi_spread_1_idx
        int i_radius = <int>radius
        np.uint8_t v

        np.ndarray[np.uint8_t, ndim=3] ret = np.zeros_like(X)
        np.uint8_t[:,:,:] ret_mv = ret

        #np.ndarray[np.uint8_t, ndim=4] intermediate_ret = np.zeros_like(X)
        #np.uint8_t[:,:,:,:] intermediate_ret_mv = intermediate_ret

    if spread == 'box' or spread is True:
        # Since box spreading is separable, we will first spread the first dimension
        # and then the second
        with nogil:
            if 0:
                for n in range(N):
                    for e in range(E):
                        for i in range(dim0):
                            for j in range(dim1):   
                                v = X_mv[n,e,i,j]
                                if v == 1:
                                    for x in range(int_max(0, i-i_radius), int_min(dim0, i+i_radius+1)):
                                        intermediate_ret_mv[n,e,x,j] = 1

                for n in range(N):
                    for e in range(E):
                        for i in range(dim0):
                            for j in range(dim1):   
                                v = intermediate_ret_mv[n,e,i,j]
                                if v == 1:
                                    for y in range(int_max(0, j-i_radius), int_min(dim1, j+i_radius+1)):
                                        ret_mv[n,e,i,y] = 1

        with nogil:
            for i in range(dim0):
                lo_spread_0_idx = max(i-i_radius,0)
                hi_spread_0_idx = min(i+i_radius+1,dim0)
                for j in range(dim1):   
                    lo_spread_1_idx = max(j-i_radius,0)
                    hi_spread_1_idx = min(j+i_radius+1,dim1)
                    for e in range(E):
                        v = X_mv[i,j,e]
                        if v == 1:
                            for x in range(lo_spread_0_idx, hi_spread_0_idx):
                                for y in range(lo_spread_1_idx, hi_spread_1_idx):
                                    ret_mv[x,y,e] = 1 
    return ret

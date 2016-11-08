
# cython: boundscheck=False
# cython: wraparound=False
# cython: embedsignature=True
# cython: cdivision=True
# cython: profile=False
import cython
import numpy as np
cimport numpy as np
#from cython.parallel import prange
from math import fabs
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


cdef inline DTYPE_t dabs(DTYPE_t x) nogil: 
    return x if x >= 0 else -x 

cdef inline DTYPE_t lerp(DTYPE_t a, DTYPE_t x, DTYPE_t y) nogil:
    return (1.0-a) * x + a * y

def _interp2d_4d(x, y, z, dx=None, startx=None, fill_value=None): 
    dx = dx if dx is not None else 1.0/(np.array(z.shape[2:]))
    startx = startx if startx is not None else np.zeros(2) 

    cdef:
        int N1 = z.shape[0]
        int N2 = z.shape[1]
        DTYPE_t startx0 = startx[0]
        DTYPE_t startx1 = startx[1]
        DTYPE_t dx0 = dx[0]
        DTYPE_t dx1 = dx[1]

        int sx0 = x.shape[0]
        int sx1 = x.shape[1]
        int sz0 = z.shape[2]
        int sz1 = z.shape[3]

        np.ndarray[DTYPE_t, ndim=4] _output = np.empty((N1, N2, sx0, sx1), dtype=DTYPE)
        DTYPE_t[:,:,:,:] output = _output

        DTYPE_t[:,:] x_mv = x
        DTYPE_t[:,:] y_mv = y
        DTYPE_t[:,:,:,:] z_mv = z

        int x0, x1
        DTYPE_t pz0, pz1
        DTYPE_t sz0max = sz0-1-1e-10
        DTYPE_t sz1max = sz1-1-1e-10
        int i, j, n1, n2 
        DTYPE_t a, intp
        int fill = (fill_value == None)
        DTYPE_t ctype_fill_value = 0.0
        DTYPE_t xp1, xp2

    if fill_value:
        ctype_fill_value = fill_value

    for x0 in xrange(sx0):
        for x1 in xrange(sx1):
            pz0 = startx0 + x_mv[x0, x1] / dx0
            pz1 = startx1 + y_mv[x0, x1] / dx1
            if fill:
                if pz0 < 0.0: pz0 = 0.0
                elif pz0 > sz0max: pz0 = sz0max

                if pz1 < 0.0: pz1 = 0.0
                elif pz1 > sz1max: pz1 = sz1max
            else:
                if dabs(pz0-(sz0-1)) < 0.0001:
                    pz0 = sz0max
                if dabs(pz1-(sz1-1)) < 0.0001:
                    pz1 = sz1max
        
            if 0.0 <= pz0 < sz0-1 and 0.0 <= pz1 < sz1-1:
                i = <int>pz0
                j = <int>pz1
                for n1 in xrange(N1):
                    for n2 in xrange(N2):
                        a = pz0-i
                        xp1 = lerp(a, z_mv[n1,n2,i,j], z_mv[n1,n2,i+1,j])
                        xp2 = lerp(a, z_mv[n1,n2,i,j+1], z_mv[n1,n2,i+1,j+1])
                        a = pz1-j
                        intp = lerp(a, xp1, xp2)
                        output[n1,n2,x0,x1] = intp
            else: 
                intp = ctype_fill_value
                for n1 in xrange(N1):
                    for n2 in xrange(N2):
                        output[n1,n2,x0,x1] = intp
    return _output 

def _interp2d_3d(x, y, z, dx=None, startx=None, fill_value=None): 
    dx = dx if dx is not None else 1.0/(np.array(z.shape[1:]))
    startx = startx if startx is not None else np.zeros(2) 

    cdef:
        int N = z.shape[0]
        DTYPE_t startx0 = startx[0]
        DTYPE_t startx1 = startx[1]
        DTYPE_t dx0 = dx[0]
        DTYPE_t dx1 = dx[1]

        int sx0 = x.shape[0]
        int sx1 = x.shape[1]
        int sz0 = z.shape[1]
        int sz1 = z.shape[2]

        np.ndarray[DTYPE_t, ndim=3] _output = np.empty((N, sx0, sx1), dtype=DTYPE)
        DTYPE_t[:,:,:] output = _output

        DTYPE_t[:,:] x_mv = x
        DTYPE_t[:,:] y_mv = y
        DTYPE_t[:,:,:] z_mv = z

        int x0, x1
        DTYPE_t pz0, pz1
        DTYPE_t sz0max = sz0-1-1e-10
        DTYPE_t sz1max = sz1-1-1e-10
        int i, j, n
        DTYPE_t a, intp
        int fill = (fill_value == None)
        DTYPE_t ctype_fill_value = 0.0
        DTYPE_t xp1, xp2

    if fill_value:
        ctype_fill_value = fill_value

    for x0 in xrange(sx0):
        for x1 in xrange(sx1):
            pz0 = startx0 + x_mv[x0, x1] / dx0
            pz1 = startx1 + y_mv[x0, x1] / dx1
            if fill:
                if pz0 < 0.0: pz0 = 0.0
                elif pz0 > sz0max: pz0 = sz0max

                if pz1 < 0.0: pz1 = 0.0
                elif pz1 > sz1max: pz1 = sz1max
            else:
                if dabs(pz0-(sz0-1)) < 0.0001:
                    pz0 = sz0max
                if dabs(pz1-(sz1-1)) < 0.0001:
                    pz1 = sz1max
        
            if 0.0 <= pz0 < sz0-1 and 0.0 <= pz1 < sz1-1:
                i = <int>pz0
                j = <int>pz1
                for n in xrange(N):
                    a = pz0-i
                    xp1 = lerp(a, z_mv[n,i,j], z_mv[n,i+1,j])
                    xp2 = lerp(a, z_mv[n,i,j+1], z_mv[n,i+1,j+1])
                    a = pz1-j
                    intp = lerp(a, xp1, xp2)
                    output[n,x0,x1] = intp
            else: 
                intp = ctype_fill_value
                for n in xrange(N):
                    output[n,x0,x1] = intp
    return _output 

def _interp2d_2d(x, y, z, dx=None, startx=None, fill_value=None): 
    dx = dx if dx is not None else 1.0/(np.array(z.shape))
    startx = startx if startx is not None else np.zeros(2) 

    cdef:
        DTYPE_t startx0 = startx[0]
        DTYPE_t startx1 = startx[1]
        DTYPE_t dx0 = dx[0]
        DTYPE_t dx1 = dx[1]

        int sx0 = x.shape[0]
        int sx1 = x.shape[1]
        int sz0 = z.shape[0]
        int sz1 = z.shape[1]

        np.ndarray[DTYPE_t, ndim=2] _output = np.empty((sx0, sx1), dtype=DTYPE)
        DTYPE_t[:,:] output = _output

        DTYPE_t[:,:] x_mv = x
        DTYPE_t[:,:] y_mv = y
        DTYPE_t[:,:] z_mv = z

        int x0, x1
        DTYPE_t pz0, pz1
        DTYPE_t sz0max = sz0-1-1e-10
        DTYPE_t sz1max = sz1-1-1e-10
        int i, j
        DTYPE_t a, intp
        int fill = (fill_value == None)
        DTYPE_t ctype_fill_value = 0.0
        DTYPE_t xp1, xp2

    if fill_value:
        ctype_fill_value = fill_value

    for x0 in xrange(sx0):
        for x1 in xrange(sx1):
            pz0 = startx0 + x_mv[x0, x1] / dx0
            pz1 = startx1 + y_mv[x0, x1] / dx1
            if fill:
                if pz0 < 0.0: pz0 = 0.0
                elif pz0 > sz0max: pz0 = sz0max

                if pz1 < 0.0: pz1 = 0.0
                elif pz1 > sz1max: pz1 = sz1max
            else:
                if dabs(pz0-(sz0-1)) < 0.0001:
                    pz0 = sz0max
                if dabs(pz1-(sz1-1)) < 0.0001:
                    pz1 = sz1max
        
            if 0.0 <= pz0 < sz0-1 and 0.0 <= pz1 < sz1-1:
                i = <int>pz0
                j = <int>pz1
                a = pz0-i
                xp1 = lerp(a, z_mv[i,j], z_mv[i+1,j])
                xp2 = lerp(a, z_mv[i,j+1], z_mv[i+1,j+1])
                a = pz1-j
                intp = lerp(a, xp1, xp2)
            else: 
                intp = ctype_fill_value

            output[x0,x1] = intp
    return _output 


def interp2d(x, y, z, dx=None, startx=None, fill_value=None): 
    """
    Calculates bilinear interpolated points of `z` at positions `x` and `y`.
    
    The motivation of this function is that ``scipy.interpolate.interp2d(..., kind='linear')`` produces unwanted results.

    It is also possible to perform several interpolations at the same time, by letting `z` have 3 or 4 dimensions, with the first (two) indexing the different sets of `z` values. More than 4 is not supported.

    Parameters
    ----------
    x, y : ndarray
        Points at which to interpolate data. Array of shape ``(A, B)``, where `A` and `B` are the rows and columns.
    z : ndarray
        The original array that should be interpolated. Array of size ``(A, B)`` or ``(N, A, B)`` or ``(N1, N2, A, B)``.
    dx : ndarray or None
        The distance between points in `z`. Array of size 2.
        If None, even spacing that range from 0.0 to 1.0 is assumed.
    startx : ndarray or None
        The ``(x, y)`` value corresponding to ``z[0,0]``. Array of size 2. 
    fill_value : float or None
        The value to return outside the area specified by `z`. If None, the closest value inside the area is used.

    Returns
    -------
    output : ndarray
        Array of shape ``(A, B)`` or ``(N, A, B)`` or ``(N1, N2, A, B)`` with interpolated values at positions at `x` and `y`.
    """
    

    assert x.dtype == DTYPE, "x must be of type {0}, not {1}".format(DTYPE, x.dtype)
    assert y.dtype == DTYPE, "y must be of type {0}, not {1}".format(DTYPE, y.dtype)
    assert z.dtype == DTYPE, "z must be of type {0}, not {1}".format(DTYPE, z.dtype)
    # This assert is only possible with 
    assert x.shape == y.shape, "x and y must be the same shape ({0} != {1})".format(x.shape, y.shape)
    
    if z.ndim == 4:
        return _interp2d_4d(x, y, z, dx, startx, fill_value)
    elif z.ndim == 3:
        return _interp2d_3d(x, y, z, dx, startx, fill_value)
    elif z.ndim == 2:
        return _interp2d_2d(x, y, z, dx, startx, fill_value)
    else:
        raise ValueError("z must have 2 or 3 dimensions")

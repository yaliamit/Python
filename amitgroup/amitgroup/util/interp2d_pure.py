
from math import fabs
from copy import copy
import numpy as np
eps = 1e-10

def lerp(a, x, y):
    return y * a + (1-a) * x 

class interp2d_func(object):
    def __init__(self, z, fill_value=None):
        self.z = np.copy(z)
        self.fill_value = fill_value
        self.startx = np.zeros(2)
        self.dx = np.ones(2)

    def __call__(self, x):
        return self._sample(self.startx + x/self.dx)

    def _sample(self, p):
        sx, sy = self.z.shape
        p = np.copy(p)
        if self.fill_value is None:
            p[0] = max(0.0, min(sx-1-eps, p[0]))
            p[1] = max(0.0, min(sy-1-eps, p[1]))
    
        if fabs(p[0]-(sx-1)) < 0.01:
            p[0] = sx-1-eps 
        if fabs(p[1]-(sy-1)) < 0.01:
            p[1] = sy-1-eps

        if 0.0 <= p[0] < sx-1 and 0.0 <= p[1] < sy-1:
            i, j = int(p[0]), int(p[1])
            a = p[0]-i
            xp1 = lerp(a, self.z[i,j], self.z[i+1,j])
            xp2 = lerp(a, self.z[i,j+1], self.z[i+1,j+1])
            a = p[1]-j
            intp = lerp(a, xp1, xp2)
            return intp
        else: 
            return self.fill_value 


def interp2d(xs, z, dx=None, startx=None, fill_value=None): 
    f = interp2d_func(z, fill_value)
    # TODO: Fix the dx
    f.dx = dx if dx is not None else 1.0/np.array(z.shape) #np.array([xs[1,0,0]-xs[0,0,0], xs[0,1,1]-xs[0,0,1]])
    f.startx = startx if startx is not None else np.zeros(2) 

    output = np.empty(z.shape)
    for x0 in range(xs.shape[0]):
        for x1 in range(xs.shape[1]):
            x = xs[x0, x1] 
            output[x0,x1] = f(x)
    return output 
    

    

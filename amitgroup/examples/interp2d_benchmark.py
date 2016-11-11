from __future__ import division
import timeit
import numpy as np
import amitgroup as ag
import pywt
from time import time


def run_test(shape):
    N = 10 
    setup = """
import numpy as np
import amitgroup as ag
z = np.arange(np.prod({0}), dtype=np.float64).reshape({0})
z = np.tile(z, (8, 1, 1))
imdef = ag.util.DisplacementFieldWavelet({0}, 'db4')
imdef.randomize(0.01)
x, y = imdef.meshgrid()
x1, y1 = imdef.deform_x(x, y)
    """.format(shape)

    dt = timeit.Timer("ag.util.interp2d(x1, y1, z)", setup).timeit(N)/N
    #dt = timeit.Timer("scipy.interpolate.interp2d(x, y, z, kind='linear', fill_value=0.0)", setup).timeit(N)/N

    return dt


if __name__ == '__main__':
    dt = run_test((32, 32))
    print 'interp2d sma:', 1000*dt, "ms"

    dt2 = run_test((1024, 1024))
    print 'interp2d big:', 1000*dt2, "ms"

    print 'start time:',1000*(1024*dt - dt2)/1023

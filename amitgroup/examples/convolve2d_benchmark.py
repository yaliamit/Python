from __future__ import division
import timeit

def run_test(shape):
    N = 10 
    setup = """
import numpy as np
import amitgroup as ag
import scipy.signal
x, _ = ag.io.load_mnist('training', selection=slice(100))
signal = ag.features.bedges(x)
kernel = np.ones({0})
    """.format(shape)

    dt = timeit.Timer("ag.util.convolve2d(signal, kernel, mode='same')", setup).timeit(N)/N
    #dt = timeit.Timer("scipy.interpolate.interp2d(x, y, z, kind='linear', fill_value=0.0)", setup).timeit(N)/N

    return dt


if __name__ == '__main__':
    dt = run_test((3, 3))
    print 'Kernel ones(3, 3):', 1000*dt, "ms"

    dt2 = run_test((5, 5))
    print 'Kernel ones(5, 5):', 1000*dt2, "ms"

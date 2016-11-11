from __future__ import division
import timeit
import numpy as np
import amitgroup as ag
import pywt
from time import time
import amitgroup.util.wavelet
import matplotlib.pylab as plt


def run_test(length, levels):
    setup = "x = np.arange({0})".format(length)
    N = 1 

    max_level = int(np.log2(length))

    setup = """
import pywt
import amitgroup.util.wavelet
import numpy as np
import amitgroup as ag
x = np.arange(np.prod({0})).reshape({0})
wavedec, waverec = ag.util.wavelet.daubechies_factory({0}, 'db2')
u = pywt.wavedec(x, 'db2', mode='per', level={1})
u2 = wavedec(x, levels={1})
    """.format(length, max_level)

    wavedec_pywt = timeit.Timer("u = pywt.wavedec(x, 'db2', mode='per', level=5)", setup).timeit(N)/N
    wavedec_amit = timeit.Timer("coefs = wavedec(x)", setup).timeit(N)/N 

    waverec_pywt = timeit.Timer("pywt.waverec(u, 'db2', mode='per')", setup).timeit(N)/N
    waverec_amit = timeit.Timer("A = waverec(x)", setup).timeit(N)/N

    return wavedec_pywt, wavedec_amit, waverec_pywt, waverec_amit


if __name__ == '__main__':
    #shapes = [(16, 16), (32, 32), (64, 64), (256, 256), (512, 512), (1024, 1024)][:
    shapes = [(1 << i) for i in range(3, 13+1)]
    shapes_labels = ["{0}".format(shape) for shape in shapes]
    print shapes
    data = np.zeros((len(shapes), 4))
    for i, shape in enumerate(shapes):
        max_level = int(np.log2(shape))
        for levels in range(1, max_level+1):
            row = run_test(shape, levels)
            data[i] = row

    # Plot it
    plt.semilogy(data, linewidth=2)
    plt.legend(('pywt (wavedec)', 'amitgroup (wavedec)', 'pywt (waverec)', 'amitgroup (waverec)'), loc=0)
    plt.xticks(range(len(shapes)), shapes_labels)
    plt.ylabel("Time [s]")
    plt.xlabel("Shape")
    plt.show()
            

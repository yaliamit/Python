from __future__ import division
import timeit
import numpy as np
import amitgroup as ag
import pywt
from time import time
import amitgroup.util.wavelet
import matplotlib.pylab as plt


def run_test(shape, levels):
    setup = "x = np.arange(np.prod({0})).reshape({0})".format(shape)
    N = 10
    #u = pywt.wavedec2(x, 'db2', mode='per', level=5)

    max_level = int(np.log2(max(shape)))

    setup = """
import pywt
import amitgroup.util.wavelet
import numpy as np
import amitgroup as ag
x = np.arange(np.prod({0})).reshape({0})
wavedec2, waverec2 = ag.util.wavelet.daubechies_factory({0}, 'db2')
u = pywt.wavedec2(x, 'db2', mode='per', level={1})
u2 = wavedec2(x, levels={1})
    """.format(shape, max_level)

    wavedec2_pywt = timeit.Timer("u = pywt.wavedec2(x, 'db2', mode='per', level=5)", setup).timeit(N)/N
    wavedec2_amit = timeit.Timer("coefs = wavedec2(x)", setup).timeit(N)/N 

    waverec2_pywt = timeit.Timer("pywt.waverec2(u, 'db2', mode='per')", setup).timeit(N)/N
    waverec2_amit = timeit.Timer("A = waverec2(x)", setup).timeit(N)/N

    #print "Shape: {0}, levels used: ({1})".format(shape, levels)
    #print "pywt wavedec2d:", 1000*wavedec2_pywt, "ms"
    #print "amit wavedec2d:", 1000*wavedec2_amit, "ms"
    #print "pywt waverec2d:", 1000*waverec2_pywt, "ms"
    #print
    return wavedec2_pywt, wavedec2_amit, waverec2_pywt, waverec2_amit


if __name__ == '__main__':
    #shapes = [(16, 16), (32, 32), (64, 64), (256, 256), (512, 512), (1024, 1024)][:
    shapes = [((1 << i),)*2 for i in range(3, 10+1)]
    shapes_labels = ["{0}x{1}".format(*shape) for shape in shapes]
    print shapes
    data = np.zeros((len(shapes), 4))
    for i, shape in enumerate(shapes):
        max_level = int(np.log2(max(shape)))
        for levels in range(1, max_level+1):
            row = run_test(shape, levels)
            data[i] = row

    # Plot it
    plt.semilogy(data, linewidth=2)
    plt.legend(('pywt (wavedec2)', 'amitgroup (wavedec2)', 'pywt (waverec2)', 'amitgroup (waverec2)'), loc=0)
    plt.xticks(range(len(shapes)), shapes_labels)
    plt.ylabel("Time [s]")
    plt.xlabel("Shape")
    plt.show()
            

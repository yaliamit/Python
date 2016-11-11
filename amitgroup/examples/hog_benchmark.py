from __future__ import division
import timeit


def run_test(img_size, cell_size, block_size):
    N = 10 
    setup = """
import numpy as np
import amitgroup as ag
import matplotlib.pylab as plt
import os.path
np.random.seed(0)
im = np.random.random({0})
    """.format(img_size)

    dt = timeit.Timer("ag.features.hog(im, {0}, {1})".format(cell_size, block_size), setup).timeit(N)/N

    return dt


if __name__ == '__main__':
    def test_and_print(img_size, cell_size, block_size):
        dt = run_test(img_size, cell_size, block_size)
        print 'image size: {0}, cell-size: {1}, block-size: {2} ... {3} ms'.format(img_size, cell_size, block_size, 1000*dt)

    test_and_print((128, 128), (6, 6), (2, 2))
    test_and_print((128, 128), (6, 6), (3, 3))

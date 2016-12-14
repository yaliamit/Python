
from __future__ import print_function

import sys
import os
import numpy as np

def load_dataset(pad=0,nval=10000):
    # We first define a download function, supporting both Python 2 and 3.
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)

    # We then define functions for loading MNIST images and labels.
    # For convenience, they also download the requested files if needed.
    import gzip

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(-1, 1, 28, 28)

        if (pad>0):
            new_data=np.zeros((data.shape[0],data.shape[1],data.shape[2]+2*pad,data.shape[3]+2*pad))
            new_data[:,:,pad:pad+28,pad:pad+28]=data
            data=new_data
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)

        return data / np.float32(256)

    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    # We can now download and read the training and test set images and labels.
    X_train = load_mnist_images('../Compare_new/MNIST/train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('../Compare_new/MNIST/train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('../Compare_new/MNIST/t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('../Compare_new/MNIST/t10k-labels-idx1-ubyte.gz')

    # We reserve the last 10000 training examples for validation.
    if (nval>0):
        X_train, X_val = X_train[:-nval], X_train[-nval:]
        y_train, y_val = y_train[:-nval], y_train[-nval:]
    else:
        X_val=None
        y_val=None
    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train, X_val, y_val, X_test, y_test


# ##################### Build the neural network model #######################
# This script supports three types of models. For each one, we define a
# function that takes a Theano variable representing the input and returns
# the output layer of a neural network model built in Lasagne.












from __future__ import print_function

import sys
import os
import numpy as np
import h5py

def one_hot(values,n_values=10):
    n_v = np.maximum(n_values,np.max(values) + 1)
    oh=np.float32(np.eye(n_v)[values])
    return oh



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
        data = data.reshape(-1, 28, 28, 1)

        if (pad>0):
            new_data=np.zeros((data.shape[0],data.shape[1],data.shape[2]+2*pad,data.shape[3]+2*pad))
            new_data[:,:,pad:pad+28,pad:pad+28]=data
            data=new_data
        # The inputs come as bytes, we convert them to floatX in range [0,1].
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
def get_mnist():
    tr, trl, val, vall, test, testl = load_dataset()
    trl=one_hot(trl)
    vall=one_hot(vall)
    testl=one_hot(testl)
    return (tr,trl), (val,vall), (test,testl)

def get_cifar(data_set='cifar10'):

    filename = '../_CIFAR100/'+data_set+'_train.hdf5'
    print(filename)
    f = h5py.File(filename, 'r')
    key = list(f.keys())[0]
    # Get the data
    tr = f[key]
    print('tr',tr.shape)
    key = list(f.keys())[1]
    tr_lb=f[key]
    train_data=np.float32(tr[0:45000])/255.
    train_labels=one_hot(np.int32(tr_lb[0:45000]))
    val_data=np.float32(tr[45000:])/255.
    val_labels=one_hot(np.int32(tr_lb[45000:]))
    filename = '../_CIFAR100/'+data_set+'_test.hdf5'
    f = h5py.File(filename, 'r')
    key = list(f.keys())[0]
    # Get the data
    test_data = np.float32(f[key])/255.
    key = list(f.keys())[1]
    test_labels=one_hot(np.int32(f[key]))
    return (train_data, train_labels), (val_data, val_labels), (test_data, test_labels)



def get_data(PARS):
    if ('cifar' in PARS['data_set']):
        train, val, test=get_cifar(data_set=PARS['data_set'])
    elif (PARS['data_set']=="mnist"):
        train, val, test= get_mnist()
    num_train = np.minimum(PARS['num_train'], train[0].shape[0])
    train = (train[0][0:num_train], train[1][0:num_train])
    dim = train[0].shape[1]
    PARS['nchannels'] = train[0].shape[3]
    PARS['n_classes'] = train[1].shape[1]
    print('n_classes', PARS['n_classes'], 'dim', dim, 'nchannels', PARS['nchannels'])
    return train, val, test, dim



from __future__ import absolute_import
import sys
import os
import numpy as np
from six.moves import cPickle


def load_mnist(data_path="data"):
    """Loads MNIST dataset.
    Function adapted from Lasagne library:
    https://github.com/Lasagne/Lasagne/blob/master/examples/mnist.py
    # Arguments
        data_path: path where data is stored
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    if data_path != "" and not os.path.exists(data_path):
        os.makedirs(data_path)

    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, os.path.join(data_path, filename))

    import gzip

    def load_mnist_images(filename):
        if not os.path.exists(os.path.join(data_path, filename)):
            download(filename)
        with gzip.open(os.path.join(data_path, filename), 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
            data = data.reshape(-1, 1, 28, 28).transpose(0, 1, 2, 3)
        return data / np.float32(255)

    def load_mnist_labels(filename):
        if not os.path.exists(os.path.join(data_path, filename)):
            download(filename)
        with gzip.open(os.path.join(data_path, filename), 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data

    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    return (X_train, y_train), (X_test, y_test)


def load_cifar10(data_path="data"):
    """Loads CIFAR10 dataset.
    Function adapted from F. Chollet's keras library:
    https://github.com/fchollet/keras/
    # Arguments
        data_path: path where data is / should be stored
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    if data_path != "" and not os.path.exists(data_path):
        os.makedirs(data_path)

    dirname = 'cifar-10-batches-py'
    origin = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

    def download(filename, source):
        if not os.path.exists(os.path.join(data_path, filename)):
            print("Downloading %s" % filename)
            urlretrieve(source + filename, os.path.join(data_path, filename))
            import tarfile
            tar = tarfile.open(os.path.join(data_path, filename))
            tar.extractall(data_path)
            tar.close()

    download('cifar-10-python.tar.gz', source=origin)
    path = os.path.join(data_path, dirname)
    nb_train_samples = 50000

    x_train = np.zeros((nb_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.zeros((nb_train_samples,), dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        data, labels = load_batch(fpath)
        x_train[(i - 1) * 10000: i * 10000, :, :, :] = data
        y_train[(i - 1) * 10000: i * 10000] = labels

    fpath = os.path.join(path, 'test_batch')
    x_test, y_test = load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    return (x_train, y_train), (x_test, y_test)


def load_rot_mnist(data_path="data/"):
    import zipfile
    name_file = "mnist_rotation_new.zip"
    path = os.path.join(data_path, name_file)
    n_train = 12000
    n_test = 50000
    size_img = 28
    zf = zipfile.ZipFile(path)
    test = zf.read("mnist_all_rotation_normalized_float_test.amat")
    train = zf.read("mnist_all_rotation_normalized_float_train_valid.amat")
    train = np.array(train.split(), dtype="float32")
    test = np.array(test.split(), dtype="float32")
    train = train.reshape(n_train, -1)
    test = test.reshape(n_test, -1)
    x_train = train[:, :size_img**2]
    x_test = test[:, :size_img**2]
    y_train = train[:, -1]
    y_test = test[:, -1]

    x_train = x_train.reshape(n_train, 1, size_img, size_img)
    x_test = x_test.reshape(n_test, 1, size_img, size_img)
    x_train = np.swapaxes(x_train, 2, 3)
    x_test = np.swapaxes(x_test, 2, 3)

    return (x_train, y_train), (x_test, y_test)


def load_batch(fpath, label_key='labels'):
    """Internal utility for parsing CIFAR data.
    Function extracted from F. Chollet's keras library:
    https://github.com/fchollet/keras/
    # Arguments
        fpath: path the file to parse.
        label_key: key for label data in the retrieve
            dictionary.
    # Returns
        A tuple `(data, labels)`.
    """
    f = open(fpath, 'rb')
    if sys.version_info < (3,):
        d = cPickle.load(f)
    else:
        d = cPickle.load(f, encoding='bytes')
        # decode utf8
        d_decoded = {}
        for k, v in d.items():
            d_decoded[k.decode('utf8')] = v
        d = d_decoded
    f.close()
    data = d['data']
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32)
    return data, labels

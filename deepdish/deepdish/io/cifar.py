from __future__ import division, print_function, absolute_import
import deepdish as dd
import numpy as np
import os


def load_cifar_10(section, offset=0, count=10000, marg=5, x_dtype=np.float32,
                  ret='xy'):
    assert section in ['training', 'testing']

    # TODO: This loads it from hdf5 files that I have prepared.

    # For now, only batches that won't be from several batches
    #assert count <= 10000
    assert offset % count == 0
#    assert 10000 % count == 0
    if (count < 10000):
        num_batches=1
    else:
        num_batches=count//10000
    batch_offset = 0
    data=[]
    for b in range(num_batches):
        if section == 'training':
            batch_number = offset // 10000 + 1 + b
            if batch_number > 5:
                name = None
                batch_offset = 0
                count = 0
            else:
                name = 'cifar_{}.h5'.format(batch_number)
                print(name)
                batch_offset = offset % 10000
        else:
            name = 'cifar_test.h5'
        batch_offset = offset

        if name is not None:
            data.append(dd.io.load(os.path.join(os.environ['CIFAR10_DIR'], name)))
        else:
            data = dict(data=np.empty(0), labels=np.empty(0))
    XX=[]
    YY=[]
    for d in data:
        XX.append(d['data'])
        YY.append(d['labels'])
    X = np.concatenate(XX)
    y = np.concatenate(YY)

    X0 = X[0:count,:].reshape(-1, 3, 32, 32)
    y0 = y[0:count]

    if x_dtype in [np.float16, np.float32, np.float64]:
        X0 = X0.astype(x_dtype) / 255
    elif x_dtype == np.uint8:
        pass  # Do nothing
    else:
        raise ValueError('load_cifar_10: Unsupported value for x_dtype')
    X0=X0.transpose(0,2,3,1)

    X0 = dd.util.pad(X0, (0, marg, marg, 0), value=0)
    returns = []
    if 'x' in ret:
        returns.append(X0)
    if 'y' in ret:
        returns.append(y0)

    if len(returns) == 1:
        return returns[0]
    else:
        return tuple(returns)


def load_cifar_100(section, offset=0, count=10000, marg=5, x_dtype=np.float32,
                  ret='xy'):
    assert section in ['training', 'testing']

    # TODO: This loads it from hdf5 files that I have prepared.

    # For now, only batches that won't be from several batches
    #assert count <= 10000
    assert offset % count == 0
#    assert 10000 % count == 0
    if (count < 50000):
        num_batches=1
    else:
        num_batches=count//50000
    batch_offset = 0
    data=[]
    for b in range(num_batches):
        if section == 'training':
            batch_number = offset // 50000 + 1 + b
            name = 'cifar100_train.h5' #.format(batch_number)
            print(name)
            #batch_offset = offset % 10000
        else:
            name = 'cifar100_test.h5'
        #batch_offset = offset

        if name is not None:
            data.append(dd.io.load(os.path.join(os.environ['CIFAR100_DIR'], name)))
        else:
            data = dict(data=np.empty(0), labels=np.empty(0))
    XX=[]
    YY=[]
    for d in data:
        XX.append(d['data'])
        YY.append(d['labels'])
    X = np.concatenate(XX)
    y = np.concatenate(YY)

    X0 = X[0:count,:].reshape(-1, 3, 32, 32)
    y0 = y[0:count]

    if x_dtype in [np.float16, np.float32, np.float64]:
        X0 = X0.astype(x_dtype) / 255
    elif x_dtype == np.uint8:
        pass  # Do nothing
    else:
        raise ValueError('load_cifar_10: Unsupported value for x_dtype')
    X0=X0.transpose(0,2,3,1)

    X0 = dd.util.pad(X0, (0, marg, marg, 0), value=0)
    returns = []
    if 'x' in ret:
        returns.append(X0)
    if 'y' in ret:
        returns.append(y0)

    if len(returns) == 1:
        return returns[0]
    else:
        return tuple(returns)

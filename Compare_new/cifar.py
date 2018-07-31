import deepdish as dd
import numpy as np
import os
from scipy import linalg
import h5py

def get_cifar(data_set='cifar10'):
    filename = '../_CIFAR100/' + data_set + '_train.hdf5'
    f = h5py.File(filename, 'r')
    key = list(f.keys())[0]
    tr = f[key]
    key = list(f.keys())[1]
    train_labels = f[key]
    train_data = np.float32(tr[0:50000]) / 255.
    filename = '../_CIFAR100/' + data_set + '_test.hdf5'
    f = h5py.File(filename, 'r')
    key = list(f.keys())[0]
    # Get the data
    test_data = np.float32(f[key]) / 255.
    key = list(f.keys())[1]
    test_labels = f[key]
    return train_data, train_labels, test_data, test_labels

def load_dataset(data_set,num_train=50000, num_test=10000, num_val=5000, marg=0, Train=True, white=True):
    home='../'#os.path.expanduser('~')
    os.environ['CIFAR10_DIR']=home+'_CIFAR10'
    os.environ['CIFAR100_DIR']=home+'_CIFAR100'
    train_x=None
    train_y=None
    val_x=None
    val_y=None
    Tr_x, Tr_y, te_x, te_y = get_cifar(data_set)

    # if (data_set=='cifar_100'):
    #     print('loading cifar_100')
    #     if (Train):
    #         Tr_x, Tr_y = dd.io.load_cifar_100('training', offset=0, count=50000, marg=marg)
    #     te_x, te_y = dd.io.load_cifar_100('testing', offset=0, count=10000, marg=marg)
    # else:
    #     print('loading cifar_10')
    #     if (Train):
    #         Tr_x, Tr_y = dd.io.load_cifar_10('training', offset=0, count=50000, marg=marg)
    #     te_x, te_y = dd.io.load_cifar_10('testing', offset=0, count=10000, marg=marg)

    if (Train):
        Tr_x=np.transpose(Tr_x,(0,3,1,2))
        Trr_x=Tr_x[:50000-num_val]
        Trr_y=Tr_y[:50000-num_val]
        num_train=np.minimum(num_train,50000-num_val)
        train_x=Trr_x[:num_train]
        train_y=Trr_y[:num_train]
        val_x=Tr_x[50000-num_val:]
        val_y=Tr_y[50000-num_val:]
        # if (white):
        #     regularization=.001
        #     if (os.path.isfile('comps.npy')):
        #         comps=np.load('comps.npy')
        #         mtrx=np.load('means.npy')
        #     else:
        #         trx=np.reshape(train_x,(train_x.shape[0],np.prod(train_x.shape[1:])))
        #         mtrx=np.mean(trx,axis=0,keepdims=True)
        #         trx=trx-mtrx
        #         sigma = np.dot(trx.T,trx) / trx.shape[1]
        #         U, S, V = linalg.svd(sigma)
        #         tmp = np.dot(U, np.diag(1/np.sqrt(S+regularization)))
        #         comps=np.dot(tmp, U.T)
        #         np.save('comps.npy',comps)
        #         np.save('means.npy',mtrx)
        #     tr_x=np.reshape(train_x,(train_x.shape[0],np.prod(train_x.shape[1:])))
        #     tr_x=np.dot((tr_x-mtrx),comps.T)
        #     train_x=np.reshape(tr_x,train_x.shape)
        #     tr_x=np.reshape(test_x,(test_x.shape[0],np.prod(test_x.shape[1:])))
        #     tr_x=np.dot((tr_x-mtrx),comps.T)
        #     test_x=np.reshape(tr_x,test_x.shape)
        #     tr_x=np.reshape(val_x,(val_x.shape[0],np.prod(val_x.shape[1:])))
        #     tr_x=np.dot((tr_x-mtrx),comps.T)
        #     val_x=np.reshape(tr_x,val_x.shape)

    if (Train):
        train_x=np.float32(train_x)
        train_y=np.float32(train_y)
        val_x=np.float32(val_x)
        val_y=np.float32(val_y)

    te_x=np.transpose(te_x,(0,3,1,2))
    test_x=te_x[:num_test]
    test_y=te_y[:num_test]

    return  train_x, train_y, val_x, val_y, np.float32(test_x), np.float32(test_y)
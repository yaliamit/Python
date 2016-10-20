import deepdish as dd
import numpy as np
import os
from scipy import linalg
def load_dataset(num_train=50000, num_test=10000, num_val=5000, marg=0, white=True):
    home=os.path.expanduser('~')
    os.environ['CIFAR10_DIR']=home+'/Desktop/Dropbox/Python/_CIFAR10'
    Tr_x, Tr_y = dd.io.load_cifar_10('training', offset=0, count=50000, marg=marg)
    te_x, te_y = dd.io.load_cifar_10('testing', offset=0, count=10000, marg=marg)

    Tr_x=np.transpose(Tr_x,(0,3,1,2))
    te_x=np.transpose(te_x,(0,3,1,2))
    Trr_x=Tr_x[:50000-num_val]
    Trr_y=Tr_y[:50000-num_val]
    num_train=np.minimum(num_train,50000-num_val)
    train_x=Trr_x[:num_train]
    train_y=Trr_y[:num_train]
    test_x=te_x[:num_test]
    test_y=te_y[:num_test]
    val_x=Tr_x[50000-num_val:]
    val_y=Tr_y[50000-num_val:]
    if (white):
        regularization=.001
        if (os.path.isfile('comps.npy')):
            comps=np.load('comps.npy')
            mtrx=np.load('means.npy')
        else:
            trx=np.reshape(train_x,(train_x.shape[0],np.prod(train_x.shape[1:])))
            mtrx=np.mean(trx,axis=0,keepdims=True)
            trx=trx-mtrx
            sigma = np.dot(trx.T,trx) / trx.shape[1]
            U, S, V = linalg.svd(sigma)
            tmp = np.dot(U, np.diag(1/np.sqrt(S+regularization)))
            comps=np.dot(tmp, U.T)
            np.save('comps.npy',comps)
            np.save('means.npy',mtrx)
        tr_x=np.reshape(train_x,(train_x.shape[0],np.prod(train_x.shape[1:])))
        tr_x=np.dot((tr_x-mtrx),comps.T)
        train_x=np.reshape(tr_x,train_x.shape)
        tr_x=np.reshape(test_x,(test_x.shape[0],np.prod(test_x.shape[1:])))
        tr_x=np.dot((tr_x-mtrx),comps.T)
        test_x=np.reshape(tr_x,test_x.shape)
        tr_x=np.reshape(val_x,(val_x.shape[0],np.prod(val_x.shape[1:])))
        tr_x=np.dot((tr_x-mtrx),comps.T)
        val_x=np.reshape(tr_x,val_x.shape)

    return  np.float32(train_x), np.float32(train_y), np.float32(val_x), np.float32(val_y), np.float32(test_x), np.float32(test_y)
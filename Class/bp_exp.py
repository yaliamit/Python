from __future__ import print_function

import make_net
import numpy as np
import time
import data
import sys
import theano.tensor as T
import theano
import untied_conv_mat
import manage_OUTPUT


def get_confusion_matrix(pred,y):
        num_class=np.max(y)+1
        conf_mat=np.zeros((num_class,num_class))
        for c in range(num_class):
            predy=pred[y==c,:]
            am_predy=np.argmax(predy, axis=1)
            u, counts=np.unique(am_predy, return_counts=True)
            conf_mat[c,u]=counts
        return(conf_mat)


def main_new(NETPARS):
    # Load the dataset

    np.random.seed(NETPARS['seed'])

    print("seed",NETPARS['seed'])
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test=data.get_train(NETPARS)
    num_class=len(np.unique(y_test))
    num_train=X_train.shape[0]
    num_test=X_test.shape[0]
    num_val=X_val.shape[0]
    ytr=np.zeros((y_train.shape[0],num_class))
    ytr[np.arange(0,num_train,1),y_train]=1
    ytr=ytr.astype(bool)
    nytr=np.logical_not(ytr)
    # yte=np.zeros((num_test,num_class))
    # yte[np.arange(0,num_test,1),y_test]=1
    # yv=np.zeros((num_val,num_class))
    # yv[np.arange(0,num_val,1),y_val]=1
    X_train=np.reshape(X_train,(X_train.shape[0],-1))
    X_test=np.reshape(X_test,(X_test.shape[0],-1))
    X_val=np.reshape(X_val,(X_val.shape[0],-1))

    # Prepare Theano variables for inputs and targets

    num_inputs=X_train.shape[1]
    num_hidden=NETPARS['num_hidden']
    eta=NETPARS['eta']

    std=np.sqrt(6./(num_inputs+num_hidden))
    W1=np.float32(np.random.uniform(-std,std,(num_inputs,num_hidden)))
    std=np.sqrt(6./(num_class+num_hidden))
    W2=np.float32(np.random.uniform(-std,std,(num_hidden,num_class)))
    R2=np.float32(np.random.uniform(-std,std,(num_hidden,num_class)))
    COST=np.zeros(NETPARS['num_epochs'])
    ERR=np.zeros(NETPARS['num_epochs'])
    POSITIVITY=np.zeros((NETPARS['num_epochs'],2))

    # Iterate
    for n in range(NETPARS['num_epochs']):

    # Forward pass
        x=T.matrix()
        w1=T.matrix()
        w2=T.matrix()

        h=T.dot(x,w1)
        o=T.dot(h,w2)
        #yh=T.argmax(,axis=1)
        #err=T.mean(y_train!=yhat)
        forward=theano.function(inputs=[x,w1,w2],output=[o])
        0=forward(X_train,W1,W2)
        #H=np.dot(X_train,W1)
        #O=np.dot(H,W2)
    # Classification
        yhat=np.argmax(O,axis=1)
        ERR[n]=np.mean(y_train!=yhat)
        D3=np.zeros((num_train,num_class))
    # Hinge cost
        if (NETPARS['cost']=='hinge'):
            L1=np.maximum(1-O[ytr],0)
            L2a=np.maximum(1+O[nytr],0)
            L2=np.sum(np.reshape(L2a,(-1,num_class-1)),axis=1)/(num_class-1)
            COST[n]=np.mean(L1+L2)
            # How many non-zero elements
            #cpos=np.float32(np.sum(L1>0))/num_train
            #npos=np.float32(np.sum(L2a>0))/num_train
        # Derivative of hinge cost
            D3[ytr]=-np.float32(L1>0)
            D3[nytr]=np.float32(L2a>0)/(num_class-1)
        else:
    # Inner product cost
            COST[n]=-np.sum(O[ytr])+np.sum(O[nytr])/(num_class-1)
            D3[ytr]=-1
            D3[nytr]=1./(num_class-1)


        print(n,'COST',COST[n],'ACC',1-ERR[n])
    # Positivity of R^tW and Correlation of R^t*W with error signal
        cc=np.zeros(num_train)
        if (NETPARS['use_R']):
            V=R2
        else:
            V=W2
        U=np.dot(V.T,W2)
        e,v=np.linalg.eig((U+U.T)/2)
        POSITIVITY[n,0]=np.float32(np.sum(e>0))/len(e)
        # UU=np.dot(V.T,V)
        # ee,vv=np.linalg.eig(UU)
        #print(np.vstack((e,ee)))
        for i,d in enumerate(D3):
            cc[i]=np.dot(np.dot(d,U),d)
        POSITIVITY[n,1]=np.float32(np.sum(cc>0))/num_train
        print('POSITIVITY',POSITIVITY[n,0],POSITIVITY[n,1])

    # Delta of W2
        DW2=np.zeros((num_hidden,num_class))
        for i,h in enumerate(H):
                DW2+=np.outer(h,D3[i])
        DW2=DW2/num_train
        W2-=eta*DW2

    # Propagation of error derivative to previous layer
        if (NETPARS['update_R'] and NETPARS['use_R']):
            R2-=eta*DW2
        if (NETPARS['use_R']):
            D2=np.dot(D3,R2.T)
        else:
            D2=np.dot(D3,W2.T)

    # Delta of W1
        if (NETPARS['update_1']):
            DW1=np.zeros((num_inputs,num_hidden))
            for i,x in enumerate(X_train):
                DW1+=np.outer(x,D2[i])
            DW1=DW1/num_train
            W1-=eta*DW1
        if (np.mod(n,10)==0):
            Hv=np.dot(X_val,W1)
            Ov=np.dot(Hv,W2)
            yvhat=np.argmax(Ov,axis=1)
            errv=np.mean(y_val!=yvhat)
            print(n,1-errv)

    return COST,ERR,POSITIVITY





from manage_OUTPUT import process_args as pa
parms={}
parms=pa(sys.argv,parms)
from  parse_net_pars import parse_text_file as ptf
NETPARS={}
ptf(parms['net'],NETPARS,lname='layers', dump=False)

#NETPARS={}
# NETPARS['num_hidden']=200
# NETPARS['Mnist']='mnist'
# NETPARS['num_train']=5000
# NETPARS['seed']=542567
# NETPARS['train']=True
# NETPARS['num_epochs']=500
# NETPARS['eta']=.1
# NETPARS['use_R']=False
# NETPARS['update_1']=True
# NETPARS['update_R']=False
# NETPARS['cost']='hinge'
# NETPARS['plot']=True
COST,ERR,POSITIVITY=main_new(NETPARS)

ss='out_R'+str(NETPARS['use_R'])+'_uR'+str(NETPARS['update_R'])+'_u1'+str(NETPARS['update_1'])+'.txt'

OUT=np.vstack((COST,1-ERR,POSITIVITY.T))
f=open(ss,'w')
np.savetxt(f,OUT.T,fmt='%6.3f',delimiter=' ')
f.close()
if (NETPARS['plot']):
    import pylab as py
    py.plot(1-ERR)
    py.plot(POSITIVITY)
    py.show()


from __future__ import print_function

import make_net
import numpy as np
import time
import data
import sys
import theano.tensor as T
import theano
import theano.tensor.nlinalg as Tn
from collections import OrderedDict
import untied_conv_mat
import manage_OUTPUT

def multiclass_hinge_loss_alt(predictions, targets, delta_up=1., delta_down=1., dep_fac=1.):

    num_cls = predictions.shape[1]
    if targets.ndim == predictions.ndim - 1:
        targets = theano.tensor.extra_ops.to_one_hot(targets, num_cls)
    elif targets.ndim != predictions.ndim:
        raise TypeError('rank mismatch between targets and predictions')
    corrects = predictions[targets.nonzero()]
    rest = theano.tensor.reshape(predictions[(1-targets).nonzero()],
                                 (-1, num_cls-1))
    if delta_down < 0:
        relc=corrects
        relr=dep_fac*rest/(num_cls-1)
        loss=theano.tensor.sum(relr,axis=1)-relc
    else:
        if (dep_fac>0):
            relc=theano.tensor.nnet.relu(delta_up-corrects)
            relr=dep_fac*theano.tensor.nnet.relu(delta_down+rest)/(num_cls-1)
            loss=theano.tensor.sum(relr,axis=1)+relc
    d3a=T.zeros(targets.shape)
    d3a=T.set_subtensor(d3a[targets.nonzero()],relc)
    d3a=-(d3a>0).astype(theano.config.floatX)
    d3b=T.zeros(targets.shape)
    d3b=T.set_subtensor(d3b[(1-targets).nonzero()],T.reshape(relr,(-1,)))
    d3b=(d3b>0).astype(theano.config.floatX)/(num_cls.astype(theano.config.floatX)-1.)

    d3=d3a+d3b
    return loss,relr,relc, d3

def get_confusion_matrix(pred,y):
        num_class=np.max(y)+1
        conf_mat=np.zeros((num_class,num_class))
        for c in range(num_class):
            predy=pred[y==c,:]
            am_predy=np.argmax(predy, axis=1)
            u, counts=np.unique(am_predy, return_counts=True)
            conf_mat[c,u]=counts
        return(conf_mat)


def update_sgd(pars, pards, eta):


    updates = OrderedDict()

    for p,dp in zip(pars,pards):
        updates[p]=p-eta*dp

    return updates

def setup_function(x,target_var,w1,w2,r2, NETPARS):

        eta=theano.shared(np.float32(NETPARS['eta']))
        num_train=x.shape[0].astype(theano.config.floatX)
        h=T.dot(x,w1)
        o=T.dot(h,w2)


        acost, relr, relc, d3 = multiclass_hinge_loss_alt(o,target_var)

        cost=acost.mean()
        acc = T.mean(T.eq(T.argmax(o, axis=1), target_var),
                              dtype=theano.config.floatX)


        if (NETPARS['use_R']):
            v=r2
        else:
            v=w2
        #U=T.dot(v.T,w2)
        #e,v=Tn.eig(U)
        #p0=0
        #p0=T.sum(e.nonzero()).astype(theano.config.floatX)/T.shape(e)[0].astype(theano.config.floatX)
        #cc=T.dot(T.dot(d3,U),d3.T)
        #p1=T.sum(T.diag(cc).nonzero())/num_train
        #dw2=T.zeros(w2.shape)
        dw2=T.mean(h.dimshuffle(0,1,'x')*d3.dimshuffle(0,'x',1),axis=0) #T.outer(h,d3))
        d2=T.dot(d3,v.T)
        #dw1=T.zeros(w1.shape)
        dw1=T.mean(x.dimshuffle(0,1,'x')*d2.dimshuffle(0,'x',1),axis=0) #T.outer(x,d2))
        dr2=T.zeros(dw2.shape)
        if (NETPARS['update_R']):
            dr2=dw2
        updates=update_sgd([w1,w2,r2],[dw1,dw2,dr2],eta)
        train=theano.function(inputs=[x,target_var],outputs=[o,h,cost,acc,d3,d2],
        updates=updates,name="train")

        return(train,eta)

    # Delta of W1
        if (NETPARS['update_1']):

            W1-=eta*DW1


def regular_iter(X_train,y_train,ytr,nytr,W1,W2,R2,eta,n,COST,POSITIVITY,ERR):

        num_inputs=W1.shape[0]
        num_hidden=W1.shape[1]
        num_train=X_train.shape[0]
        num_class=len(np.unique(y_train))

        H=np.dot(X_train,W1)
        O=np.dot(H,W2)
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
    #eta=NETPARS['eta']

    std=np.sqrt(6./(num_inputs+num_hidden))
    W1=np.float32(np.random.uniform(-std,std,(num_inputs,num_hidden)))
    std=np.sqrt(6./(num_class+num_hidden))
    W2=np.float32(np.random.uniform(-std,std,(num_hidden,num_class)))
    R2=np.float32(np.random.uniform(-std,std,(num_hidden,num_class)))
    COST=np.zeros(NETPARS['num_epochs'])
    ERR=np.zeros(NETPARS['num_epochs'])
    POSITIVITY=np.zeros((NETPARS['num_epochs'],2))

    input_var=T.matrix('input')
    w1=theano.shared(W1)
    w2=theano.shared(W2)
    r2=theano.shared(R2)
    target_var = T.ivector('target')

    train,eta=setup_function(input_var,target_var,w1,w2,r2,NETPARS)

    # Iterate
    for n in range(NETPARS['num_epochs']):

        [O,H,cost,acc,D3,D2]=train(X_train,y_train)
        print(cost,acc)
    # Forward pass

        # regular_iter(X_train,y_train,ytr,nytr,W1,W2,R2,eta,n,COST,POSITIVITY,ERR)
        #
        # if (np.mod(n,10)==0):
        #     Hv=np.dot(X_val,W1)
        #     Ov=np.dot(Hv,W2)
        #     yvhat=np.argmax(Ov,axis=1)
        #     errv=np.mean(y_val!=yvhat)
        #     print(n,1-errv)


    return COST,ERR,POSITIVITY





from manage_OUTPUT import process_args as pa
parms={}
parms=pa(sys.argv,parms)
from  parse_net_pars import parse_text_file as ptf
NETPARS={}
ptf(parms['net'],NETPARS,lname='layers', dump=False)

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


from __future__ import print_function

import sys
from collections import OrderedDict

import numpy as np
import pylab as py
import theano
import theano.tensor as T
import theano.tensor.nlinalg as Tn

import data
from run_compare import adamloc
from source.run_class import iterate_minibatches_new


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
    d3b=dep_fac*(d3b>0).astype(theano.config.floatX)/(num_cls.astype(theano.config.floatX)-1.)

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

def setup_function(x,target_var,w1,r1,fw1,w2,r2,fw2,eta, NETPARS,train=True):



        num_train=x.shape[0].astype(theano.config.floatX)
        # Hidden layer
        h=T.dot(x,w1)
        if ('tanh' in NETPARS):
            h=T.nnet.sigmoid(h)
        o=T.dot(h,w2)


        acost, relr, relc, d3 = multiclass_hinge_loss_alt(o,target_var)

        cost=acost.mean()
        ans=T.argmax(o,axis=1)
        num_cls = o.shape[1]
        if (train):
            O=theano.tensor.extra_ops.to_one_hot(target_var, num_cls)
        else:
            O=theano.tensor.extra_ops.to_one_hot(target_var, num_cls)
        #d3=-theano.tensor.extra_ops.to_one_hot(target_var,num_cls)
        acc = T.mean(T.eq(ans, target_var),
                              dtype=theano.config.floatX)




        if (NETPARS['use_R']):
            v2=r2
        else:
            v2=w2
        # Angle between R and W
        U=T.dot(v2.T,w2)
        U=(U+U.T)/2


        # Delta of W2
        dw2=T.mean(h.dimshuffle(0,1,'x')*d3.dimshuffle(0,'x',1),axis=0)
        # Propagate error backwards
        d2=T.dot(d3,v2.T)
        dw1=T.zeros(w1.shape)
        if (NETPARS['update_1']):
            dw1=T.mean(x.dimshuffle(0,1,'x')*d2.dimshuffle(0,'x',1),axis=0)

        # Propagate output backwards
        fh=T.dot(O,fw2.T)
        if ('tanh' in NETPARS):
            fh=T.nnet.sigmoid(fh)
        #dfw2=T.zeros(w2.shape)
        dfw2=T.mean(h.dimshuffle(0,1,'x')*O.dimshuffle(0,'x',1),axis=0)
        fx=T.dot(fh,fw1.T)
        #fx=T.dot(O,fw1.T)
        if ('tanh' in NETPARS):
            fx=T.nnet.sigmoid(fx)
        fx=(1.*fx+0.*x)
        # Update W1
        dfw1=T.mean(x.dimshuffle(0,1,'x')*fh.dimshuffle(0,'x',1),axis=0)
        #dfw1=T.mean(x.dimshuffle(0,1,'x')*O.dimshuffle(0,'x',1),axis=0)


        dr1=T.zeros(dw1.shape)
        dr2=T.zeros(dw2.shape)
        # Update R2
        if (NETPARS['update_R']):
            dr2=dw2
            dr1=dw1
        if (NETPARS['update']=='sgd'):
            updates=update_sgd([w1,r1,fw1,w2,r2,fw2],[dw1,dr1,dfw1,dw2,dr2,dfw2],eta)
        else:
            updates=adamloc([dw1,dr1,dfw1,dw2,dr2,dfw2],[w1,r1,fw1,w2,r2,fw2],learning_rate=eta)
        if (train):
            fn=theano.function(inputs=[x,target_var],outputs=[o,h,cost,acc,d3,d2,U,fx],
                updates=updates,name="train")
        else:
            fn=theano.function(inputs=[x,target_var],outputs=[cost,acc,fx],updates=None)
        return(fn)








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



    X_train=np.reshape(X_train,(X_train.shape[0],-1))
    X_test=np.reshape(X_test,(X_test.shape[0],-1))
    X_val=np.reshape(X_val,(X_val.shape[0],-1))

    # Prepare Theano variables for inputs and targets

    num_inputs=X_train.shape[1]
    num_hidden=NETPARS['num_hidden']

    std=np.sqrt(6./(num_inputs+num_hidden))
    W1=np.float32(np.random.uniform(-std,std,(num_inputs,num_hidden)))
    fW1=np.float32(np.random.uniform(-std,std,(num_inputs,num_hidden)))
    #fW1=np.float32(np.random.uniform(-std,std,(num_inputs,num_class)))

    R1=np.float32(np.random.uniform(-std,std,(num_inputs,num_hidden)))
    std=np.sqrt(6./(num_class+num_hidden))
    W2=np.float32(np.random.uniform(-std,std,(num_hidden,num_class)))
    fW2=np.float32(np.random.uniform(-std,std,(num_hidden,num_class)))
    R2=np.float32(np.random.uniform(-std,std,(num_hidden,num_class)))
    COST=np.zeros(NETPARS['num_epochs'])
    ERR=np.zeros(NETPARS['num_epochs'])
    POSITIVITY=np.zeros(NETPARS['num_epochs'])

    input_var=T.matrix('input')
    w1=theano.shared(W1)
    fw1=theano.shared(fW1)
    r1=theano.shared(R1)
    w2=theano.shared(W2)
    fw2=theano.shared(fW2)
    r2=theano.shared(R2)
    target_var = T.ivector('target')
    eta=theano.shared(np.float32(NETPARS['eta']))
    train_fn=setup_function(input_var,target_var,w1,r1,fw1,w2,r2,fw2,eta,NETPARS,train=True)
    test_fn=setup_function(input_var,target_var,w1,r1,fw1,w2,r2,fw2,eta,NETPARS,train=False)

    # Iterate
    BX=[]
    for n in range(NETPARS['num_epochs']):

        CST=0
        ACC=0
        num_batches=0
        for batches,batch in enumerate(iterate_minibatches_new(X_train, y_train, NETPARS['batch_size'], shuffle=True)):
            inputs,targets = batch
            [O,H,cost,acc,D3,D2,U,bx]=train_fn(inputs,targets)
            ACC+=acc
            CST+=cost
            num_batches+=1
            #e,v=np.linalg.eig(U)
            #POSITIVITY[n]=np.sum(e>0)/np.float32(len(e))
        print('CORR',n,CST/num_batches,ACC/num_batches) #,POSITIVITY[n])
        COST[n]=CST/num_batches
        ERR[n]=1-ACC/num_batches

    COST=0
    ACC=0
    num_batches=0
    for batches,batch in enumerate(iterate_minibatches_new(X_train, y_train, NETPARS['batch_size'], shuffle=False)):
            inputs,targets = batch
            [cost,acc,bx]=test_fn(inputs,targets)
            ACC+=acc
            COST+=cost
            num_batches+=1
            BX.append(bx)
    print('FINAL CORR',COST/num_batches,ACC/num_batches)
    BXX=np.concatenate(BX)
    ii=np.random.randint(0,num_train,10)


    #py.ion()
    for i in ii:
        fig=py.figure()
        inp=np.reshape(X_train[i,],(28,28))
        out=np.reshape(BXX[i,],(28,28))
        fig.add_subplot(1,2,1)
        py.imshow(inp,cmap='gray')
        fig.add_subplot(1,2,2)
        py.imshow(out,cmap='gray')
        py.show()

    print("DONE")
    return COST,ERR,POSITIVITY



from manage_OUTPUT import process_args as pa

from  parse_net_pars import parse_text_file as ptf

parms={}
parms=pa(sys.argv,parms)
NETPARS={}
ptf(parms['net'],NETPARS,lname='layers', dump=False)

for key, value in parms.iteritems():
        NETPARS[key]=parms[key]

COST,ERR,POSITIVITY=main_new(NETPARS)

print('DONE')
sys.stdout.flush()

if (NETPARS['plot']):

    py.plot(1-ERR)
    py.plot(POSITIVITY)
    py.show()



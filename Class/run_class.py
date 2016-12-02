from __future__ import print_function


import mnist
import make_net
import numpy as np
import theano
import theano.typed_list
import theano.tensor as T
import time
import lasagne
import run_compare
import data
import sys
import scipy.stats

def get_confusion_matrix(pred,y):
        num_class=np.max(y)+1
        conf_mat=np.zeros((num_class,num_class))
        for c in range(num_class):
            predy=pred[y==c,:]
            am_predy=np.argmax(predy, axis=1)
            u, counts=np.unique(am_predy, return_counts=True)
            conf_mat[c,u]=counts
        return(conf_mat)



def iterate_minibatches_new(inputs, targets, batchsize, shuffle=False):
    if (type(inputs) is not list):
        assert len(inputs) == len(targets)
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt], targets[excerpt]
    else:
        num_data=inputs[0].shape[0]
        if shuffle:
            indices = np.arange(num_data)
        for start_idx in range(0, num_data - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            out=[]
            for inp in inputs:
                out.append(inp[excerpt])
            yield out, targets[excerpt]


def iterate_on_batches(func,X,y,batch_size,typ='Test',fac=False, agg=False, seq=0, network=None, pars=None):
    if (len(X)==0):
        return(0,0)
    shuffle=False
    if (typ=='Train'):
        shuffle=True
    if (pars is not None and 'trans' in pars):
        X=data.do_rands(X,pars,insert=True)
        ll=len(X)
        X=np.concatenate(X,axis=0)
        y=np.tile(y,ll)
        if (typ=='Test'):
            fac=ll
    err=acc=0
    pred=[]
    grad=None

    for batches,batch in enumerate(iterate_minibatches_new(X, y, batch_size, shuffle=shuffle)):
        inputs,targets = batch
        if (type(func) is not list):
            tout=func(inputs,targets)
        else:
            for f in func:
                tout=f(inputs,targets)
        # Information on gradient magnitude
        acc += tout[1]; err += tout[0]
        if (fac or agg):
            pred.append(tout[2])
        #loss.append(tout[3])

    # if len(tout)==4:
    #    pr=np.sum(np.abs(np.array(network.W.eval())))
    # # Aggregating over angles.

    if (fac):
        pred0=np.concatenate(pred)
        pred1=np.reshape(pred0,(fac,pred0.shape[0]/fac)+pred0.shape[1:])
        pred_m=np.max(pred1,axis=2)
        pred_am=np.argmax(pred1,axis=2)
        pred_amm=np.argmax(pred_m,axis=0)
        pred2=np.mean(pred1,axis=0)
        #pred2=np.max(pred1,axis=0)
        ypred=np.argmax(pred2,axis=1)
        ypreda=pred_am[tuple(pred_amm),range(pred_am.shape[1])]
        newacc=np.mean(ypred==y[:len(y)/fac])
        newacca=np.mean(ypreda==y[:len(y)/fac])

        pred=pred1[np.int32(np.floor((len(pred1)-1)/2))]
        print("Mean over different rotations",newacc)
        print("Maxmax over different rotations",newacca)
        # preed=np.zeros(pred2.shape)
        # for j in range(pred1.shape[1]):
        #     ee=np.zeros(pred1.shape[0])
        #     for i in range(pred1.shape[0]):
        #         ee=scipy.stats.entropy(pred1[i,j,:])
        #         ie=np.argmin(ee)
        #         preed[j,:]=pred1[ie,j,:]
        # ypreed=np.argmax(preed,axis=1)
        # neewacc=np.mean(ypreed==y[:len(y)/fac])
        # print("Min ent over rotations",neewacc)
    if (agg and not fac):
        pred=np.concatenate(pred)

    print("Final results:")
    print(typ+" loss:\t\t\t{:.6f}".format(err / (batches+1)))
    print(typ+" acc:\t\t\t{:.6f}".format(acc / (batches+1)))


    sys.stdout.flush()
    return(acc,batches, pred, grad, fac)


def main_new(NETPARS):
    # Load the dataset
    np.random.seed(NETPARS['seed'])
    batch_size=NETPARS['batch_size']
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test=data.get_train(NETPARS)

    # Prepare Theano variables for inputs and targets
    if (type(X_train) is not list):
        input_var =  T.tensor4('inputs')
        if (X_train is not None):
            dims=X_train.shape[1:]
        else:
            dims=X_test.shape[1:]
    else:
        if (X_train is not None):
            dims=X_train[0].shape[1:]
        else:
            dims=X_test[0].shape[1:]
        if (theano.config.device=='cpu'):
            input_var=theano.typed_list.TypedListType(theano.tensor.dtensor4)()
            for j in range(len(X_train)):
                theano.typed_list.append(input_var,T.dtensor4())
        else:
            input_var=theano.typed_list.TypedListType(theano.tensor.ftensor4)()
            length=theano.typed_list.length(input_var)
            for j in range(len(X_train)):
                theano.typed_list.append(input_var,T.ftensor4())
        if ('trans' in NETPARS and NETPARS['simple_augmentation'] is False):
            NETPARS['layers'][0]['augment']=len(X_train)
    target_var = T.ivector('target')

    NETPARS['layers'][0]['dimx']=dims[1]
    NETPARS['layers'][0]['dimy']=dims[2]
    NETPARS['layers'][0]['num_input_channels']=dims[0]


    print("Building model and compiling functions...")
    # Create neural network model (depending on first command line parameter)
    network = make_net.build_cnn_on_pars(input_var,NETPARS)
    step=T.iscalar()
    params=lasagne.layers.get_all_params(network)
    num_steps=len(params)
    #step=0
    pp=T.tensor4()
    if (NETPARS['train']):
        if ('seq' in NETPARS):
            train_fn, eta=run_compare.setup_function_seq(network,NETPARS,input_var,target_var,step,Train=True)
        else:
            train_fn,eta=run_compare.setup_function(network,NETPARS,input_var,target_var,Train=True)

    val_fn,dummy=run_compare.setup_function(network,NETPARS,input_var,target_var,Train=False)

    NETPARS=make_net.make_file_from_params(network,NETPARS)
    # Iterate over epochs:
    eta_p=run_compare.eta_params(network)
    if (NETPARS['train'] and NETPARS['num_epochs']>0):
        print("Starting training...","Training set size:",X_train.shape[0])
        mod_eta=True
        #if NETPARS['update']!='adam':
        #    mod_eta=True
        for epoch in range(NETPARS['num_epochs']):
            # In each epoch, do a full pass over the training data:
            start_time = time.time()
            print("eta",eta.get_value())
            out_tr=iterate_on_batches(train_fn,X_train,y_train,batch_size,typ='Train',network=network,pars=NETPARS)
            out_te=iterate_on_batches(val_fn,X_val,y_val,batch_size,typ='Val')
            if (NETPARS['adapt_eta']):
                eta_p.update(out_te[0],out_te[1],network,eta,mod_eta=mod_eta)
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, NETPARS['num_epochs'], time.time() - start_time))
            # Save params in intermediate steps
            if (np.mod(epoch,10)==0 and epoch>0):
                params = lasagne.layers.get_all_param_values(network)
                np.save(NETPARS['output_net'],params)
        # Final save of params
        params = lasagne.layers.get_all_param_values(network)
        NETPARS['eta_current']=eta.get_value()
        prev_epochs=0
        if ('prev_epochs' in NETPARS):
            prev_epochs=NETPARS['prev_epochs']
        NETPARS['prev_epochs']=prev_epochs+epoch+1
        NETPARS=make_net.make_file_from_params(network,NETPARS)
        if (NETPARS['num_epochs']>0):
            if (NETPARS['adapt_eta']):
                print('Updating best params to network')
                np.save(NETPARS['output_net'],eta_p.best_params)
                lasagne.layers.set_all_param_values(network,eta_p.best_params)
            else:
                np.save(NETPARS['output_net'],params)

    # After training, we compute and print the test error
    fac=0
    if (type(NETPARS['simple_augmentation']) is int):
         fac=NETPARS['simple_augmentation']
    out_test=iterate_on_batches(val_fn,X_test,y_test,batch_size,typ='Test',agg=True,fac=fac,pars=NETPARS)


    conf_mat=get_confusion_matrix(out_test[2],y_test[0:(out_test[2]).shape[0]])
    print(conf_mat)
    #if (fac==0):
    fac=1
    out_test=out_test+(y_test[0:len(y_test)/fac],)
    if (NETPARS['train']):
        iterate_on_batches(val_fn,X_train,y_train,batch_size,typ='Post-train',fac=False, agg=True) #NETPARS['simple_augmentation'])


    return(NETPARS,out_test)

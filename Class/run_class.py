from __future__ import print_function

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
import scipy.sparse as sp
import densesparse
import newdensesparse
import newdense
import Conv2dLayerR

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


def iterate_on_batches(func,X,y,batch_size,typ='Test',fac=False, agg=False, network=None, pars=None):
    if (len(X)==0):
        return(0,0)
    shuffle=False
    if (typ=='Train'):
        shuffle=True
    # Randomized augmentation at each batch step instead of for the whole data set at the beginning

    if (pars is not None and type(pars) is dict and 'trans' in pars and pars['trans']['repeat']):
        X=data.do_rands(X,pars,pars['trans']['insert'])
        ll=1
        if (type(X) is list):
            ll=len(X)
            X=np.concatenate(X,axis=0)
            y=np.tile(y,ll)
        if (typ=='Test'):
            fac=ll
    curr_class=-1
    if (pars is not None and (type(pars) is dict and 'one' in pars)):
        curr_class=pars['one']
    elif type(pars) is int:
        curr_class=pars
    if (curr_class>=0):
        yy=y+1
        yy[yy>curr_class]=0
    else:
        yy=y
    err=acc=0
    pred=[]
    grad=None

    for batches,batch in enumerate(iterate_minibatches_new(X, yy, batch_size, shuffle=shuffle)):
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
        newacc=np.mean(ypred==yy[:len(yy)/fac])
        newacca=np.mean(ypreda==yy[:len(yy)/fac])

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

# Prepare the functions that will make the full matrix corresponding to a convolutional layer given its input
def get_matrix_func(network,input_var):

    layers=lasagne.layers.get_all_layers(network)

    GET_CONV=[]
    for l in layers:
        if (l.name is not None):
            if ('conv' in l.name and 'S' in l.name):
                    dims=l.input_shape[1:]
                    input_v=T.tensor4()
                    new_layer=lasagne.layers.InputLayer(shape=(None,dims[0],dims[1],dims[2]),input_var=input_v)
                    new_net=lasagne.layers.Conv2DLayer(new_layer, num_filters=l.num_filters, filter_size=l.filter_size,
                                nonlinearity=lasagne.nonlinearities.linear,pad=l.pad,
                                W=l.W)
                    out=lasagne.layers.get_output(new_net)
                    get_conv=theano.function([input_v], out, updates=None)
                    GET_CONV.append(get_conv)
                    # Get the matrix corresponding to the feedback R
                    if ('R' in l.name):
                        dims=l.input_shape[1:]
                        input_v=T.tensor4()
                        new_layer=lasagne.layers.InputLayer(shape=(None,dims[0],dims[1],dims[2]),input_var=input_v)
                        new_net=lasagne.layers.Conv2DLayer(new_layer, num_filters=l.num_filters, filter_size=l.filter_size,
                                    nonlinearity=lasagne.nonlinearities.linear,pad=l.pad,
                                    W=l.R)
                        out=lasagne.layers.get_output(new_net)
                        get_conv=theano.function([input_v], out, updates=None)
                        GET_CONV.append(get_conv)

    return GET_CONV

# Apply the functions to get the matrices and turn them into sparse matrices
def get_matrix(l,gl):
                    dims=l.input_shape[1:]
                    nda=np.prod(dims[1:])

                    t=0
                    # Create the identity matrix that will extract the sparse matrix corresponding
                    # to the linear map defined by conv.
                    rr=tuple(np.arange(0,dims[0],4))+(dims[0],)
                    yy=0
                    for r in range(len(rr)-1): #range(dims[0]):
                        print('r',r)
                        nd=(rr[r+1]-rr[r])*nda
                        XX=np.zeros((nd,)+dims)
                        t=0
                        for i in np.arange(rr[r],rr[r+1],1):
                            for j in range(dims[1]):
                                for k in range(dims[2]):
                                    XX[t,i,j,k]=1.
                                    t+=1
                        XX=np.float32(XX)
                        YY=gl(XX)
                        YY=np.reshape(YY,(YY.shape[0],np.prod(YY.shape[1:])))
                        yy+=np.prod(YY.shape)
                        print(yy)
                        if (r==0):
                            csc=sp.csc_matrix(YY)
                        else:
                            cscr=sp.csc_matrix(YY)
                            csc=sp.vstack([csc,cscr],format='csc')

                    print('Sparsity:',yy,len(csc.data),np.float32(len(csc.data))/yy)
                    return(csc)

# Put the sparse matrices in a new network and write it out.
def apply_get_matrix(network,GET_CONV, NETPARS):

    layers=lasagne.layers.get_all_layers(network)
    il=0
    SP=[]
    for l in layers:
        if (l.name is not None):
            if ('conv' in l.name and 'S' in l.name):
                SP.append(get_matrix(l,GET_CONV[il]))
                il+=1
                if ('R' in l.name):
                    SP.append(get_matrix(l,GET_CONV[il]))
                    il+=1

    # Now make a network which is a copy of the original but with dense sparse layers intstead of conv layers,
    # initialized with the collected sparse matrices.
    layer_list=[]
    t=0
    for l in layers:
        if 'input' in l.name:
            layer_list.append(lasagne.layers.InputLayer(l.shape,input_var=l.input_var,name=l.name))
        elif 'drop' in l.name:
            layer_list.append(lasagne.layers.DropoutLayer(layer_list[-1],p=l.input_layers[1].p, name=l.name))
        elif 'pool' in l.name:
            layer_list.append(lasagne.layers.Pool2DLayer(layer_list[-1],pool_size=l.pool_size, name=l.name))
        elif 'dense' in l.name:
            layer_list.append(lasagne.layers.DenseLayer(layer_list[-1],num_units=l.num_units,nonlinearity=l.nonlinearity,W=l.W, b=None, name=l.name))
        elif 'batch' in l.name:
            layer_list.append(lasagne.layers.BatchNormLayer(layer_list[-1],name=l.name,beta=l.beta,gamma=l.gamma,mean=l.mean,inv_std=l.inv_std))
        elif 'newdens' in l.name:
            lpars=lasagne.layers.get_all_param_values(l)
            # W=l.W.eval()
            # R=l.R.eval()
            # Wzero=l.Wzero.eval()
            # Rzero=l.Rzero.eval()
            layer_list.append(newdense.NewDenseLayer(layer_list[-1],num_units=l.num_units,
                                            nonlinearity=l.nonlinearity,W=lpars[-4],R=lpars[-3], Wzero=lpars[-2], Rzero=lpars[-1], b=None, name=l.name))
        elif 'conv' in l.name:
            # Sparse
            if 'S' in l.name:
                num_units=SP[t].shape[1]
                W = theano.shared(SP[t])
                t+=1
                # Also separate R
                if 'R' in l.name:
                    R=theano.shared(SP[t])
                    t=t+1
                    layer_list.append(newdensesparse.SparseDenseLayer(layer_list[-1],num_units=num_units,
                                            W=W,R=R, b=None,nonlinearity=l.nonlinearity,name='sparseR'+str(t)))
            # JUst sparse
                else:
                    layer_list.append(densesparse.SparseDenseLayer(layer_list[-1],num_units=num_units,
                                            W=W, b=None,nonlinearity=l.nonlinearity,name='sparse'+str(t)))
                # Reshape for subsequent pooling
                #shp=l.output_shape[1:]
                #layer_list.append(lasagne.layers.reshape(layer_list[-1],([0],)+shp,name='reshape'+str(t)))
            # Stays conv
            else:
                # Separate R
                if 'R' in l.name:
                    layer_list.append(Conv2dLayerR.Conv2DLayerR(layer_list[-1], pad=l.pad, num_filters=l.num_filters, filter_size=l.filter_size,
                                nonlinearity=l.nonlinearity,W=l.W,R=l.R,prob=l.prob,name=l.name, b=None))
                else:
                    layer_list.append(lasagne.layers.Conv2DLayer(layer_list[-1],num_filters=l.num_filters,name=l.name,
                                                             filter_size=l.filter_size,pad=l.pad,W=l.W,nonlinearity=l.nonlinearity))


    new_net=layer_list[-1]
    print('Done making sparse network')
    spparams=lasagne.layers.get_all_param_values(new_net)
    params=lasagne.layers.get_all_param_values(network)
    PARS=NETPARS.copy()
    ss=str.split(PARS['output_net'],'/')
    ss[-1]='sp'+ss[-1]
    PARS['output_net']='/'.join(ss)
    np.save(PARS['output_net'],spparams)
    make_net.make_file_from_params(new_net,PARS)
    print("done writing it")



def main_new(NETPARS):
    # Load the dataset
    np.random.seed(NETPARS['seed'])
    batch_size=NETPARS['batch_size']
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test=data.get_train(NETPARS)
    num_class=len(np.unique(y_test))
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
            # Sequential update of layers. Not interesting.
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
            pars=None
            if ('one' in NETPARS):
                pars=NETPARS['one']
            out_te=iterate_on_batches(val_fn,X_val,y_val,batch_size,typ='Val',pars=pars)

            if (epoch>0 and 'one' in NETPARS and np.mod(epoch,NETPARS['num_epochs']/num_class)==0):
                NETPARS['one']+=1
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
            # if (NETPARS['adapt_eta']):
            #     print('Updating best params to network')
            #     np.save(NETPARS['output_net'],eta_p.best_params)
            #     lasagne.layers.set_all_param_values(network,eta_p.best_params)
            # else:
            np.save(NETPARS['output_net'],params)

    if ('write_sparse' in NETPARS and NETPARS['write_sparse']):
            GET_CONV=get_matrix_func(network,input_var)
            apply_get_matrix(network,GET_CONV,NETPARS)
    # After training, we compute and print the test error
    fac=0
    if (type(NETPARS['simple_augmentation']) is int):
         fac=NETPARS['simple_augmentation']
    out_test=iterate_on_batches(val_fn,X_test,y_test,batch_size,typ='Test',agg=True,fac=fac,pars=NETPARS)


    conf_mat=get_confusion_matrix(out_test[2],y_test[0:(out_test[2]).shape[0]])
    np.set_printoptions(precision=4, suppress=True)
    print(conf_mat)
    #if (fac==0):
    fac=1
    out_test=out_test+(y_test[0:len(y_test)/fac],)
    if (NETPARS['train']):
        iterate_on_batches(val_fn,X_train,y_train,batch_size,typ='Post-train',fac=False, agg=True) #NETPARS['simple_augmentation'])


    return(NETPARS,out_test)

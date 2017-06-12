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
import untied_conv_mat

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

    labels=np.unique(targets)
    num_class=max(labels)+1

    max_class_per_batch=200


    if (type(inputs) is not list):
            num_data=len(targets)
            assert len(inputs) == len(targets)
            if shuffle:
                indices = np.arange(len(inputs))
                np.random.shuffle(indices)
            for start_idx in range(0, len(inputs), batchsize):
                if shuffle:
                    if (start_idx+batchsize>num_data):
                        excerpt=  indices[start_idx:num_data]
                    else:
                        excerpt = indices[start_idx:start_idx + batchsize]
                else:
                    if (start_idx+batchsize>num_data):
                        excerpt=  slice(start_idx,num_data)
                    else:
                        excerpt = slice(start_idx, start_idx + batchsize)
                yield inputs[excerpt], targets[excerpt]
    else:
        num_data=inputs[0].shape[0]
        if shuffle:
            indices = np.arange(num_data)
        #for start_idx in range(0, num_data - batchsize + 1, batchsize):
        for start_idx in range(0, num_data, batchsize):
            if shuffle:
                if (start_idx+batchsize>num_data):
                    excerpt=  indices[start_idx:num_data]
                else:
                    excerpt = indices[start_idx:start_idx + batchsize]
            else:
                if (start_idx+batchsize>num_data):
                    excerpt=  slice(start_idx,num_data)
                else:
                    excerpt = slice(start_idx, start_idx + batchsize)
            out=[]
            for inp in inputs:
                out.append(inp[excerpt])
            yield out, targets[excerpt]


def print_W_and_Wgrad_info(network,tout,pred,yy):
        np.set_printoptions(precision=4,linewidth=130)
        layers=lasagne.layers.get_all_layers(network)
        d=0

        grad=np.zeros(((len(tout)-4),8))
        t=0
        d=0
        for l in layers:
            if ('dens' in l.name or 'conv' in l.name):
                WW=np.array(l.W.eval())
                grad[t,0]=np.mean(WW)
                grad[t,1]=np.std(WW)
                sp=np.mean(WW==0)
                if (hasattr(l,'R') and ('conv' in l.name or l.Rzero.shape[0]>1)):
                    grad[t,2]=np.mean(np.array(l.R.eval()))
                    grad[t,3]=np.std(np.array(l.R.eval()))
                grad[t,4]=np.mean(tout[d+4])
                grad[t,5]=np.std(tout[d+4])
                d=d+1
                if (hasattr(l,'R')and ('conv' in l.name or l.Rzero.shape[0]>1)):
                    grad[t,6]=np.mean(tout[d+4])
                    grad[t,7]=np.std(tout[d+4])
                    d=d+1
                print('zz:',l.name,':',grad[t,],':',sp)
                t=t+1

        yp=np.argmax(pred,axis=1)
        print(np.mean(np.max(pred[yp==yy],axis=1)),np.std(np.max(pred[yp==yy],axis=1)))
        print(np.mean(np.max(pred[yp!=yy],axis=1)),np.std(np.max(pred[yp!=yy],axis=1)))


def iterate_on_batches(func,X,y,batch_size,typ='Test',fac=False, agg=False, network=None, pars=None, iter=None):
    if (len(X)==0):
        return(0,0)
    shuffle=False
    if (typ=='Train'):
        shuffle=True
    # Randomized augmentation at each batch step instead of for the whole data set at the beginning
    if (typ!='Val' and pars is not None and type(pars) is dict and 'trans' in pars and pars['trans']['repeat']):
        X=data.do_rands(X,pars,pars['trans']['insert'])
        ll=1
        if (type(X) is list):
            ll=len(X)
            X=np.concatenate(X,axis=0)
            y=np.tile(y,ll)
        if (typ=='Test'):
            fac=ll

    yy=y
    err=acc=0
    pred=[]

    for batches,batch in enumerate(iterate_minibatches_new(X, yy, batch_size, shuffle=shuffle)):
        inputs,targets = batch
        if (type(func) is not list):
            tout=func(inputs,targets)
        else:
            for f in func:
                tout=f(inputs,targets)
        # Information on gradient magnitude
        acc += tout[1]; err += tout[0]

        pred.append(tout[2])


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

    if (not fac):
        pred=np.concatenate(pred)

    print("Final results:")
    print(typ+" loss:\t\t\t{:.6f}".format(err / (batches+1)))
    print(typ+" acc:\t\t\t{:.6f}".format(acc / (batches+1)))

    if (pars is not None and 'Classes' in pars and pars['Classes'] is not None and typ!='Train'):
        lcl=pars['Done_Classes']+pars['Classes']
        yind=np.in1d(yy,lcl)
        yp=np.argmax(pred[:,lcl],axis=1)
        yp=np.array(lcl)[yp]
        errc=np.mean(yy[yind]!=yp[yind])
        print('Number of learned classes',len(lcl))
        print('Error on learned classes', errc)
    #if (network is not None and iter is not None and np.mod(iter,10)==0):
    #    print_W_and_Wgrad_info(network,tout,pred,yy)

    sys.stdout.flush()



    return(err,batches, pred, fac)



def main_new(NETPARS):
    # Load the dataset

    np.random.seed(NETPARS['seed'])
    batch_size=NETPARS['batch_size']
    print("seed",NETPARS['seed'])
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test=data.get_train(NETPARS)


    if ('Classes' not in NETPARS):
        NETPARS['Classes']=None
    elif (NETPARS['Classes']!='None' and NETPARS['Classes'] is not None):
        lcl=list()
        if ('Done_Classes' in NETPARS and NETPARS['Done_Classes']!='None'):
            lcl=lcl+list(np.int32(NETPARS['Done_Classes']))
        lcl=lcl+list(np.int32(NETPARS['Classes']))
        if (NETPARS['train']):
            yind=np.in1d(y_train,lcl)
            y_train=y_train[yind]
            X_train=X_train[yind,:]
            yind=np.in1d(y_val,lcl)
            y_val=y_val[yind]
            X_val=X_val[yind,:]
        yind=np.in1d(y_test,lcl)
        y_test=y_test[yind]
        X_test=X_test[yind,:]

    num_class=len(np.unique(y_test))
    if ('num_class' in NETPARS):
        num_class=NETPARS['num_class']['num_class']

    print("num_class", num_class)
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
    network = make_net.build_cnn_on_pars(input_var,NETPARS,num_class=num_class)

    step=T.iscalar()
    params=lasagne.layers.get_all_params(network)

    if (NETPARS['train']):
        if ('seq' in NETPARS):
            # Sequential update of layers. Not interesting.
            train_fn, eta=run_compare.setup_function_seq(network,NETPARS,input_var,target_var,step,Train=True)
        else:
            train_fn,eta, tclasses=run_compare.setup_function(network,NETPARS,input_var,target_var,Train=True)

    val_fn,dummy, tdummy=run_compare.setup_function(network,NETPARS,input_var,target_var,Train=False)

    NETPARS=make_net.make_file_from_params(network,NETPARS)
    if (NETPARS['Classes'] is not None):
        NETPARS['Done_Classes']=list()
        NETPARS['Classes']=list(NETPARS['Classes'])
    # Iterate over epochs:
    eta_p=run_compare.eta_params(network)
    curr_sched=0
    if (NETPARS['train'] and NETPARS['num_epochs']>0):
        icl=0
        print("Starting training...","Training set size:",X_train.shape[0])
        mod_eta=True
        #if NETPARS['update']!='adam':
        #    mod_eta=True
        ty_train=y_train; tX_train=X_train; ty_val=y_val; tX_val=X_val
        for epoch in range(NETPARS['num_epochs']):
            if ('num_class' in NETPARS and np.mod(epoch,NETPARS['num_class']['class_epoch'])==0):
                out_te=iterate_on_batches(val_fn,X_train,y_train,batch_size,typ='Val',pars=NETPARS)
                bdel=NETPARS['num_class']['batch_size']
                if icl>0:
                    NETPARS['Done_Classes']=list(np.unique(NETPARS['Done_Classes']+NETPARS['Classes']))
                else:
                    NETPARS['Done_Classes']=list()
                if  NETPARS['num_class']['det']:
                    NETPARS['Classes']=list(np.arange(np.mod(icl,num_class),np.mod(icl+bdel-1,num_class)+1,1))
                else:
                    ii=range(num_class)
                    np.random.shuffle(ii)
                    NETPARS['Classes']=list(np.sort(ii[0:bdel]))
                print('icl',icl,'Classes',NETPARS['Classes'])
                print(NETPARS['Done_Classes'])
                value=np.array(network.W.eval())
                if (icl==0 and NETPARS['num_class']['first']):
                    std=np.sqrt(6./(value.shape[0]+100))
                    value=np.float32(np.zeros(value.shape))
                    value[:,NETPARS['Classes']]=np.float32(np.random.uniform(-std,std,(value.shape[0],bdel)))
                    network.W.set_value(value)
                #else:
                #    std=np.std(value[:,0:icl])/10
                icl+=bdel
                cl_temp=np.zeros((1,NETPARS['num_class']['num_class']),dtype=np.float32)
                cl_temp[0,NETPARS['Classes']]=1
                tclasses.set_value(np.array(cl_temp))
                if ('sub' in NETPARS['num_class']):
                    yind=np.in1d(y_train,NETPARS['Classes'])
                    ty_train=y_train[yind]
                    tX_train=X_train[yind]
                    yind=np.in1d(y_val,NETPARS['Classes'])
                    ty_val=y_val[yind]
                    tX_val=X_val[yind]


            # In each epoch, do a full pass over the training data:
            start_time = time.time()
            print("eta",eta.get_value())
            out_tr=iterate_on_batches(train_fn,tX_train,ty_train,batch_size,typ='Train',network=network,pars=NETPARS,iter=epoch)
            #pars=None
            out_te=iterate_on_batches(val_fn,tX_val,ty_val,batch_size,typ='Val',pars=NETPARS)

            if ('eta_schedule' in NETPARS):
                sc=NETPARS['eta_schedule']
                if (curr_sched<len(sc)):
                    for i in np.arange(curr_sched,len(sc),2):
                        if (sc[i]<epoch):
                            curr_sched+=2
                            eta.set_value(np.float32(sc[i+1]))
            if (NETPARS['adapt_eta']):
                eta_p.update(out_te[0],out_te[1],network,eta,mod_eta=mod_eta)
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, NETPARS['num_epochs'], time.time() - start_time))
            # Save params in intermediate steps
            # if (np.mod(epoch,10)==0 and epoch>0):
            #     params = lasagne.layers.get_all_param_values(network)
            #     np.save(NETPARS['output_net'],params)



        # Final save of params
        params = lasagne.layers.get_all_param_values(network)
        NETPARS['eta_current']=eta.get_value()
        prev_epochs=0
        if ('prev_epochs' in NETPARS):
            prev_epochs=NETPARS['prev_epochs']
        NETPARS['prev_epochs']=prev_epochs+epoch+1
        NETPARS=make_net.make_file_from_params(network,NETPARS)
        if (NETPARS['num_epochs']>0):
            np.save(NETPARS['output_net'],params)

    if ('write_sparse' in NETPARS and NETPARS['write_sparse']):
            GET_CONV=untied_conv_mat.get_matrix_func(network,input_var,NETPARS)
            untied_conv_mat.apply_get_matrix(network,GET_CONV,NETPARS)
    # After training, we compute and print the test error
    fac=0
    if (type(NETPARS['simple_augmentation']) is int):
         fac=NETPARS['simple_augmentation']
    out_test=iterate_on_batches(val_fn,X_test,y_test,batch_size,typ='Test',agg=True,fac=fac,pars=None)


    conf_mat=get_confusion_matrix(out_test[2],y_test[0:(out_test[2]).shape[0]])
    np.set_printoptions(precision=4, suppress=True)
    print(conf_mat)
    #if (fac==0):
    fac=1
    out_test=out_test+(y_test[0:len(y_test)/fac],)
    if (NETPARS['train']):
        iterate_on_batches(val_fn,X_train,y_train,batch_size,typ='Post-train',fac=False, agg=True, pars=None) #NETPARS['simple_augmentation'])


    return(NETPARS,out_test)

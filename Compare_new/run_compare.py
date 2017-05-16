import mnist
import numpy as np
import theano
import theano.sparse as sparse
import scipy.sparse as sp
import theano.tensor as T
import time
import lasagne
import os
import make_net
import data
from theano.tensor.shared_randomstreams import RandomStreams

from collections import OrderedDict


def adamloc(loss_or_grads, params, learning_rate=0.001, beta1=0.9,
         beta2=0.999, epsilon=1e-8):

    all_grads = lasagne.updates.get_or_compute_grads(loss_or_grads, params)
    t_prev = theano.shared(lasagne.utils.floatX(0.))
    updates = OrderedDict()

    # Using theano constant to prevent upcasting of float32
    one = T.constant(1)

    t = t_prev + 1
    a_t = learning_rate*T.sqrt(one-beta2**t)/(one-beta1**t)

    for param, g_t in zip(params, all_grads):
        value = param.get_value(borrow=True)
        tens=True
        if (type(param) is theano.sparse.sharedvar.SparseTensorSharedVariable):
            tens=False
            m_prev = theano.shared(sp.csc_matrix((np.zeros(len(value.data),dtype=np.float32),value.indices,value.indptr),value.shape))
            v_prev = theano.shared(sp.csc_matrix((np.zeros(len(value.data),dtype=np.float32),value.indices,value.indptr),value.shape))
        else:
            m_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),broadcastable=param.broadcastable)
            v_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),broadcastable=param.broadcastable)

        m_t = beta1*m_prev + (one-beta1)*g_t
        #v_t = beta2*v_prev + (one-beta2)*g_t**2
        v_t= beta2*v_prev + (one-beta2)*g_t*g_t
        if (tens):
            step = a_t*m_t/(T.sqrt(v_t) + epsilon)
        else:
            step = a_t*m_t
            ss=sparse.sqrt(v_t)
            #ss=sparse.structured_add_s_v(ss,np.reshape(np.float32(epsilon),(1,1)))
            ss=ss+np.float32(epsilon)*sparse.basic.sp_ones_like(ss)
            ss=sparse.structured_pow(ss,-1.)
            step = step*ss

        updates[m_prev] = m_t
        updates[v_prev] = v_t
        updates[param] = param - step

    updates[t_prev] = t
    return updates

def clip_w(updates,params,clipt):
    for i,p in enumerate(params):
        updates[p]=theano.tensor.clip(updates[p],-clipt[i],clipt[i])
    return updates




def multiclass_hinge_loss_alt(predictions, targets, delta_up=1., delta_down=1., dep_fac=1.):

    num_cls = predictions.shape[1]
    if targets.ndim == predictions.ndim - 1:
        targets = theano.tensor.extra_ops.to_one_hot(targets, num_cls)
    elif targets.ndim != predictions.ndim:
        raise TypeError('rank mismatch between targets and predictions')
    corrects = predictions[targets.nonzero()]
    rest = theano.tensor.reshape(predictions[(1-targets).nonzero()],
                                 (-1, num_cls-1))
    if (dep_fac>0):
        relc=theano.tensor.nnet.relu(delta_up-corrects)
        relr=dep_fac*theano.tensor.nnet.relu(delta_down+rest)/(num_cls-1)
        # sftfac=1.
        # relc=theano.tensor.nnet.softplus(sftfac*(delta_up-corrects))/sftfac
        # relr=dep_fac*theano.tensor.nnet.softplus(sftfac*(delta_down+rest))/(sftfac*(num_cls-1))
        loss=theano.tensor.sum(relr,axis=1)+relc
    else:
        restm=theano.tensor.max(rest,axis=1)
        err=delta_up-corrects+restm
        loss=theano.tensor.nnet.relu(err)
    return loss



def get_training_data(NETPARS):
    X_train_in, y_train_in, X_val_in, y_val_in, X_test_in, y_test_in = data.get_train(NETPARS)
    num_train=NETPARS['num_train']
    if (num_train==0):
        num_train=np.shape(y_train_in)[0]
    NETPARS['num_train']=num_train
    if ('trans' in NETPARS):
        X_train_in=mnist.rotate_dataset_rand(X_train_in[:num_train,],NETPARS)
        X_val_in=mnist.rotate_dataset_rand(X_val_in[:num_train],NETPARS)
        X_test_in=mnist.rotate_dataset_rand(X_test_in[:num_train],NETPARS)

    #X_train_r=rotate_dataset(X_train,12,num_train)
    #X_val_r=rotate_dataset(X_val,12,np.shape(X_val)[0])
    X_train,  X_train_c, y_train=data.create_paired_data_set(NETPARS,X_train_in, y_train_in, num_train,reps=NETPARS['reps'])
    X_val, X_val_c, y_val = data.create_paired_data_set(NETPARS,X_val_in, y_val_in, num_train)
    X_test, X_test_c, y_test = data.create_paired_data_set(NETPARS,X_test_in, y_test_in, num_train)
    X_test1=X_test_f=y_test_f=y_label=None
    if (NETPARS['Mnist']):
        X_test1, X_test_f, y_test_f, y_label = data.create_paired_data_set_with_fonts(X_test_in, y_test_in, 10000)
    print('Training set size',X_train.shape[0],'Validation set size',X_val.shape[0])
    return     X_train, X_train_c, y_train, X_val, X_val_c, y_val, X_test, X_test_c, y_test, X_test1, X_test_f, y_test_f, y_label


class eta_params:

    def __init__(self,network):
        self.best_params=lasagne.layers.get_all_param_values(network)
        self.bad_count=0
        self.good_count=0
        self.best_e=100
        self.val_e_old=100

    # train_err is the loss - it should be decreasing.
    def update(self,train_err,train_batches,network,eta,mod_eta=False):
        val_e=train_err/(train_batches+1)
        #val_e=val_err/val_batches
        if (val_e < self.best_e):
            print('updating best')
            self.best_params=lasagne.layers.get_all_param_values(network)
            self.best_e=val_e

        if (mod_eta):
            # loss increased increase bad_count reset good_count
            if (self.val_e_old <= val_e):
                self.bad_count+=1
                self.good_count=0
            else:
                self.good_count+=1
            # If there has been a good stretch reset bad_count to 0

            if (self.good_count==3):
                self.bad_count=0
            # Twice increase in loss or Drastic increase in bloss exit
            if (self.bad_count>0 or self.val_e_old < .7*val_e):
                # Put in the best parameters on training data
                #lasagne.layers.set_all_param_values(network,self.best_params)
                #print('resetting to best params')
                eta.set_value(eta.get_value()*np.float32(.7))
                self.bad_count=0
                val_e=self.best_e
            self.val_e_old=val_e


def iterate_on_batches(func,X_list,y,batch_size,typ='Test',y_lab=None):
    shuffle=False
    if (typ=='Train'):
        shuffle=True
    err=acc = 0
    corrs=[]
    yy=[]
    for batches,batch in enumerate(iterate_minibatches_new(X_list, y, batch_size, shuffle=shuffle)):
        inputs, targets = batch
        tout=func(inputs, targets)
        acc += tout[1]; err += tout[0]
        if (typ=='Test1'):
            corrs.append(np.reshape(tout[2],(10,-1)))
        else:
            corrs.append(tout[2])
            yy.append(targets)
            #print('Non Same', np.mean(tout[2][targets==0]),np.std(tout[2][targets==0]))
            #print('Same', np.mean(tout[2][targets==1]),np.std(tout[2][targets==1]))



    if (typ=='Test1'):
        CORRS=np.vstack(corrs)
        yii=np.argmax(CORRS,axis=1)
    else:
        CORRS=np.hstack(corrs)
        YY=np.hstack(yy)
        print('Non Same', np.mean(CORRS[YY==0]),np.std(CORRS[YY==0]))
        print('Same', np.mean(CORRS[YY==1]),np.std(CORRS[YY==1]))
    print("Final results:")
    print(typ+" loss:\t\t\t{:.6f}".format(err / (batches+1)))
    if (typ != 'Test1'):
        print(typ+" acc:\t\t\t{:.6f}".format(acc / (batches+1)))
    else:
        print("  test acc font:\t\t\t{:.6f}".format(np.double(np.sum(yii==y_lab)) / len(yii)))
    return(err,batches)

# Setup the theano function that computes the loss from the network output
def setup_function(network,NETPARS,input_var,target_var,Train=True,loss_type='class'):


        params = lasagne.layers.get_all_params(network, trainable=True)
        spen=[]
        spe=theano.shared(0.)
        print('params',params)
        if ('reg_param_features' in NETPARS and NETPARS['reg_param_features']>0):
            reg_p = NETPARS['reg_param_features'] #theano.shared(np.array(NETPARS['reg_param'], dtype=theano.config.floatX))
            layers=lasagne.layers.get_all_layers(network)
            for l in layers:
                if (l.name is not None):
                    if ('pool' in l.name):
                      if (NETPARS['sparse_layer'] in l.name):
                        out=lasagne.layers.get_output(l)
                        spe+=T.mean(out)
                        nonz=T.mean(T.lt(out,.0001))
                        spen.append(nonz)
            spe=reg_p*spe
            spen.append(spe)
        elif ('reg_param_weights' in NETPARS and NETPARS['reg_param_weights']>0):
            reg_p = NETPARS['reg_param_weights']
            for p in params:
                if ('dense' in p.name):
                    spe+=T.mean(T.abs_(p))*reg_p
            spen.append(spe)
        if (Train):
            pred = lasagne.layers.get_output(network)
        else:
            pred = lasagne.layers.get_output(network, deterministic=True)
        gloss=[]
        if (loss_type=='class'): # Output is a probability vector on classes.
            pred = T.flatten(pred,outdim=2)
            if ('hinge' not in NETPARS or not NETPARS['hinge']):
                aloss = lasagne.objectives.categorical_crossentropy(pred, target_var)
            else:
                delta_down=1.
                dep_fac=1.
                if ('hinge_down' in NETPARS):
                    delta_down=NETPARS['hinge_down']
                if ('dep_fac' in NETPARS):
                    dep_fac=NETPARS['dep_fac']
                aloss= multiclass_hinge_loss_alt(pred,target_var,delta_up=NETPARS['hinge'],delta_down=delta_down,dep_fac=dep_fac)

            loss = aloss.mean()
            loss=loss+spe
            acc = T.mean(T.eq(T.argmax(pred, axis=1), target_var),
                          dtype=theano.config.floatX)

            layers=lasagne.layers.get_all_layers(network)
            for l in layers:
                   if ('dens' in l.name or 'conv' in l.name):
                       if (hasattr(l,'W')):
                           gloss.append(T.grad(loss,l.W))
                       if (hasattr(l,'R') and ('conv' in l.name or l.Rzero.shape[0]>1)):
                           gloss.append(T.grad(loss,l.R))

            # Instead of randomly dropping inputs drop updates on some subsets of weights.
            # This is a more severe drop because it doesn't update this subset at all in that step.
        else: #Output is two tensors that need to be compared through correlation
            pred=correlation(pred[0,],pred[1,])
            loss=T.mean(T.square(pred-target_var))
            acc = T.mean(T.eq(pred>NETPARS['thresh'], target_var),
                      dtype=theano.config.floatX)

        eta=None
        if (Train):
            eta = theano.shared(np.array(NETPARS['eta_init'], dtype=theano.config.floatX))
            if ('update' in NETPARS):
                if (NETPARS['update']=='adam'):
                    print('Using adam to update timestep')
                    updates=adamloc(loss, params, learning_rate=eta, beta1=0.9,beta2=0.999,epsilon=1e-08)
                elif (NETPARS['update']=='nestorov'):
                    print('Using Nestorov momentum')
                    updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=eta, momentum=0.9)
                elif (NETPARS['update']=='momentum'):
                    print('momentum')
                    updates = lasagne.updates.momentum(loss, params, learning_rate=eta, momentum=0.9)
                elif (NETPARS['update']=='sgd'):
                    updates = lasagne.updates.sgd(loss, params, learning_rate=eta)
                if ('clip' in NETPARS):
                    updates=clip_w(updates,params,clipt=NETPARS['clip'])

            else:
                updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=eta, momentum=0.9)
        else:
            updates=None
        #XX=T.grad(loss,input_var)

        if ('reg_param_weights' in NETPARS and Train):
            train_fn = theano.function([input_var,target_var], [loss, acc, pred]+spen, updates=updates)
        else:
            train_fn = theano.function([input_var,target_var], [loss, acc, pred, aloss]+ gloss, updates=updates)
            #train_fn = theano.function(inp, [loss, acc], updates=updates)


        return(train_fn,eta)


def setup_function_seq(network,NETPARS,input_var,target_var,step,Train=True,loss_type='class'):


        params=lasagne.layers.get_all_params(network,trainable='True')
        eta = theano.shared(np.array(NETPARS['eta_init'], dtype=theano.config.floatX))

        #for pp in reverse(params):
        if (Train):
            pred = lasagne.layers.get_output(network)
        else:
            pred = lasagne.layers.get_output(network, deterministic=True)

        pred = T.flatten(pred,outdim=2)
        aloss = lasagne.objectives.categorical_crossentropy(pred, target_var)
        loss = aloss.mean()

        acc = T.mean(T.eq(T.argmax(pred, axis=1), target_var),
                      dtype=theano.config.floatX)

        if (Train):
            train_fn=[]
            for pp in reversed(params):
                print(pp)
                gloss=[]
                gloss.append(T.grad(loss,pp))
                updates=lasagne.updates.adam(gloss, [pp,], learning_rate=0.001, beta1=0.9,beta2=0.999,epsilon=1e-08)
                train_fn.append(theano.function([input_var,target_var],[loss,acc],updates=updates))
        else:
            train_fn = theano.function([input_var,target_var], [loss, acc], updates=None)
            #train_fn = theano.function(inp, [loss, acc], updates=updates)


        return(train_fn,eta)


# Correlation function for the correlation loss.
def correlation(input1,input2):

    n=T.shape(input1)
    n0=n[0]
    n1=n[1]

    s0=T.std(input1,axis=1,keepdims=True)#.reshape((n0,1)),reps=n1)
    s1=T.std(input2,axis=1,keepdims=True)#.reshape((n0,1)),reps=n1)
    m0=T.mean(input1,axis=1,keepdims=True)
    m1=T.mean(input2,axis=1,keepdims=True)

    corr=T.sum(((input1-m0)/s0)*((input2-m1)/s1), axis=1)/n1

    corr=(corr+np.float32(1.))/np.float32(2.)
    corr=T.reshape(corr,(n0,))
    return corr

def iterate_minibatches_new(inputs, targets, batchsize, shuffle=False):

    if shuffle:
        indices = np.arange(len(inputs[0]))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs[0]) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        out=[]
        for inp in inputs:
            out.append(inp[excerpt])
        yield out, targets[excerpt]


def main_new(NETPARS):

    # Load the dataset
    if ('seed' in NETPARS):
       np.random.seed(NETPARS['seed'])
    print("Loading data...")
    X_train, X_train_c, y_train, X_val, X_val_c, y_val, X_test, X_test_c, y_test, \
            X_test1, X_test_f, y_test_f, y_label=get_training_data(NETPARS)

    dimx=X_train.shape[2]
    dimy=X_train.shape[3]
    target_var = T.fvector('target')

    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
    if (theano.config.device=='gpu'):
                input_var=theano.typed_list.TypedListType(theano.tensor.ftensor4)()
                for j in range(2):
                    theano.typed_list.append(input_var,T.ftensor4())
    else:
                input_var=theano.typed_list.TypedListType(theano.tensor.dtensor4)()
                for j in range(2):
                    theano.typed_list.append(input_var,T.dtensor4())

    NETPARS['layers'][0]['dimx']=dimx
    NETPARS['layers'][0]['dimy']=dimy
    NETPARS['layers'][0]['num_input_channels']=X_train.shape[1]
    network=make_net.build_cnn_on_pars(input_var, NETPARS)
    # if (NETPARS['network']=='regular'):
    #         network = compare_net.build_cnn_new_conv(NETPARS, input_var1, input_var2, dimx=dimx, dimy=dimy)
    # else:
    #     network = compare_net.build_cnn_new_conv_deep(NETPARS, input_var1, input_var2, dimx=dimx, dimy=dimy)
    if (os.path.isfile(NETPARS['net']+'.npy') and NETPARS['use_existing']):
        spars=np.load(NETPARS['net']+'.npy')
        lasagne.layers.set_all_param_values(network,spars)
    if (NETPARS['train']):
        train_fn,eta=setup_function(network,NETPARS,input_var,target_var,Train=True,loss_type='corr')
    val_fn,dummy=setup_function(network,NETPARS,input_var,target_var,Train=False,loss_type='corr')



    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    eta_p=eta_params(network)
    if (NETPARS['train']):
        for epoch in range(NETPARS['num_epochs']):
            start_time = time.time()
            print("eta",eta.get_value())
            # Do run on training data updating parameters
            out_tr=iterate_on_batches(train_fn,[X_train,X_train_c],y_train,NETPARS['batch_size'],typ='Train')
            # Run on validation data not updating
            out_te=iterate_on_batches(val_fn,[X_val,X_val_c],y_val,NETPARS['batch_size'],typ='Val')
            # Update time step if required.
            if (NETPARS['adapt_eta']):
                eta_p.update(out_tr[0],out_tr[1],network,eta)
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, NETPARS['num_epochs'], time.time() - start_time))
            # Save network during training.
            if (np.mod(epoch,10)==0 and epoch>0):
                params = lasagne.layers.get_all_param_values(network)
                np.save(NETPARS['output_net'],params)
        # Save Final network if training actually took place.
        if (NETPARS['num_epochs']>0):
            params = lasagne.layers.get_all_param_values(network)
            np.save(NETPARS['output_net'],params)
    iterate_on_batches(val_fn,[X_train,X_train_c],y_train,NETPARS['batch_size'],typ='Post-train')
    iterate_on_batches(val_fn,[X_test,X_test_c],y_test,NETPARS['batch_size'],typ='Test')


    if (X_test1 is not None):
        iterate_on_batches(val_fn,[X_test1,X_test_f],y_test_f,NETPARS['batch_size'],typ='Test1',y_lab=y_label)



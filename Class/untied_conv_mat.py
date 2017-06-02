import numpy as np
import theano
import theano.typed_list
import theano.tensor as T
import lasagne
import newdense
import Conv2dLayerR
import make_net


# Prepare the functions that will make the full matrix corresponding to a convolutional layer given its input
def get_matrix_func(network,input_var,NETPARS):

    layers=lasagne.layers.get_all_layers(network)

    GET_CONV=[]
    for l in layers:
        if (l.name is not None):
            if ('conv' in l.name and l.name in NETPARS['sparsify']):
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
                            csc=YY
                            #csc=sp.csc_matrix(YY)
                        else:
                            #cscr=sp.csc_matrix(YY)
                            #csc=sp.vstack([csc,cscr],format='csc')
                            cscr=YY
                            csc=np.vstack([csc,cscr])
                    #print('Sparsity:',yy,len(csc.data),np.floatX(len(csc.data))/yy)
                    return(csc)

# Put the sparse matrices in a new network and write it out.
def apply_get_matrix(network,GET_CONV, NETPARS):

    layers=lasagne.layers.get_all_layers(network)
    il=0
    SP=[]
    for l in layers:
        if (l.name is not None):
            if ('conv' in l.name and 'sparsify' in NETPARS and l.name in NETPARS['sparsify']):
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
            layer_list.append(lasagne.layers.Pool2DLayer(layer_list[-1],pool_size=l.pool_size, stride=l.stride, pad=l.pad, name=l.name,mode=l.mode))
        elif 'dense' in l.name:
            layer_list.append(lasagne.layers.DenseLayer(layer_list[-1],num_units=l.num_units,nonlinearity=l.nonlinearity,W=l.W, b=None, name=l.name))
        elif 'batch' in l.name:
            layer_list.append(lasagne.layers.BatchNormLayer(layer_list[-1],name=l.name,beta=l.beta,gamma=l.gamma,mean=l.mean,inv_std=l.inv_std))
        elif 'newdens' in l.name:
            lpars=lasagne.layers.get_all_param_values(l)
            #Rz, Wz are not preserved so no need to do them correctly, only when reading in the network.
            # Put in a random initial R with the correct sparsity.
            input_dim=lpars[-1].shape[0]
            num_units=lpars[-1].shape[1]
            std=np.sqrt(6./(input_dim+num_units))
            R=theano.shared((np.float32(np.random.uniform(-std,std,(input_dim,num_units))))*(lpars[-1]>0))
            layer_list.append(newdense.NewDenseLayer(layer_list[-1],num_units=l.num_units,prob=l.prob,
                                            nonlinearity=l.nonlinearity,W=lpars[-2],R=R, b=None, name=l.name))
        elif 'conv' in l.name:
            if 'sparsify' in NETPARS and l.name in NETPARS['sparsify']:
                num_units=SP[t].shape[1]
                input_dim=SP[t].shape[0]
                std=np.sqrt(6./(input_dim+num_units))
                #W = theano.shared(SP[t])
                W=theano.shared((np.float32(np.random.uniform(-std,std,(input_dim,num_units))))*(SP[t]>0))
                t+=1
                # Also separate R
                if 'R' in l.name:
                    R=theano.shared((np.float32(np.random.uniform(-std,std,(input_dim,num_units))))*(SP[t]>0))
                    # Record all non-zero entries of SP i.e. the ones corresponding to the conv filters.
                    t=t+1
                    layer_list.append(newdense.NewDenseLayer(layer_list[-1],num_units=num_units,prob=l.prob,
                                            W=W,R=R, b=None,nonlinearity=l.nonlinearity,name='newdens'+str(t)))
            # JUst sparse
                else:
                    layer_list.append(newdense.NewDenseLayer(layer_list[-1],num_units=num_units,
                                            W=W, b=None,Rzero=np.float32(np.ones((1,1))), prob=(1.,-1.),nonlinearity=l.nonlinearity,name='newdens'+str(t)))
                # Reshape for subsequent pooling
                shp=l.output_shape[1:]
                layer_list.append(lasagne.layers.reshape(layer_list[-1],([0],)+shp,name='reshape'+str(t)))
            # Stays conv
            else:
                # Separate R
                if 'R' in l.name:
                    WW=l.W.eval()
                    RR=l.R.eval()
                    layer_list.append(Conv2dLayerR.Conv2DLayerR(layer_list[-1], pad=l.pad, num_filters=l.num_filters, filter_size=l.filter_size,
                                nonlinearity=l.nonlinearity,W=WW,R=RR,prob=l.prob,name=l.name, b=None))
                else:
                    layer_list.append(lasagne.layers.Conv2DLayer(layer_list[-1],num_filters=l.num_filters,name=l.name,
                                                             filter_size=l.filter_size,pad=l.pad,W=l.W,b=l.b,nonlinearity=l.nonlinearity))


    new_net=layer_list[-1]
    print('Done making sparse network')
    spparams=lasagne.layers.get_all_param_values(new_net)
    PARS=NETPARS.copy()
    ss=str.split(PARS['output_net'],'/')
    ss[-1]='sp'+ss[-1]
    PARS['output_net']='/'.join(ss)
    np.save(PARS['output_net'],spparams)
    make_net.make_file_from_params(new_net,PARS)
    NETPARS['output_net']=PARS['output_net']
    print("done writing it")


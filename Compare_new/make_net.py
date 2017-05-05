from __future__ import print_function

import numpy as np
import lasagne
import theano.tensor as T
import parse_net_pars
import os
import newdense
import newdensesparse
import lasagne.init
import lasagne.utils
import Conv2dLayerR
import theano.tensor.nnet
import densesparse
import make_net
import scipy.sparse as sp
# Experimenting with git
# Experimenting again
# And again
# And again
# And again
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

def rect_sym(x):
    """Rectify activation function :math:`\\varphi(x) = \\max(0, x)`
    Parameters
    ----------
    x : float32
        The activation (the summed, weighted input of a neuron).
    Returns
    -------
    float32
        The output of the rectify function applied to the activation.
    """
    y=theano.tensor.nnet.relu(x+1)-1
    z=-theano.tensor.nnet.relu(-y+1)+1
    return z


class Trunc_Normal(lasagne.init.Initializer):
    """Sample initial weights from the Gaussian distribution.
    Initial weight parameters are sampled from N(mean, std).
    Parameters
    ----------
    std : float
        Std of initial parameters.
    mean : float
        Mean of initial parameters.
    """
    def __init__(self, mean=0.0, std=0.1, low=None, high=None):
        self.std = std
        self.mean = mean
        self.low=low
        self.high=high

    def sample(self, shape):
        A=lasagne.random.get_rng().normal(self.mean, self.std, size=shape)
        #return lasagne.utils(floatX(get_rng().normal(self.mean, self.std, size=shape)))
        if (self.low is not None):
             A=np.maximum(A,self.low)
        if (self.high is not None):
             A=np.minimum(A,self.high)
        A=lasagne.utils.floatX(A)
        return A


class SclLayer(lasagne.layers.Layer):
    def __init__(self, incoming, scale=1., **kwargs):
        super(SclLayer, self).__init__(incoming, **kwargs)
        self.fac=scale

    def get_output_for(self,input,deterministic=False,**kwargs):
        return(input*T.constant(self.fac))


class BnoiseLayer(lasagne.layers.Layer):

    def __init__(self, incoming, p=0.5, rescale=True, **kwargs):
        super(BnoiseLayer, self).__init__(incoming, **kwargs)
        self._srng = RandomStreams(lasagne.random.get_rng().randint(1, 2147462579))
        self.p = p
        self.rescale = rescale


    def get_output_for(self, input, deterministic=False, **kwargs):
        """
        Parameters
        ----------
        input : tensor
            output from the previous layer
        deterministic : bool
            If true dropout and scaling is disabled, see notes
        """
        if deterministic or self.p == 0:
            return T.ones_like(input)
        else:
            # Using theano constant to prevent upcasting
            one = T.constant(1)

            retain_prob = one - self.p
            fac=one
            if self.rescale:
                 fac /= retain_prob

            # use nonsymbolic shape for dropout mask if possible
            input_shape = self.input_shape
            if any(s is None for s in input_shape):
                input_shape = input.shape
            return self._srng.binomial(input_shape, p=retain_prob,dtype=input.dtype)*fac

class sftlayer(lasagne.layers.Layer):
    def __init__(self,incoming, **kwargs):
        super(sftlayer,self).__init__(incoming,**kwargs)

    def get_output_for(self, input, **kwargs):

        aa=T.transpose(input,(0,2,3,1))
        nn=aa.shape
        aa=T.reshape(aa,(aa.shape[0]*aa.shape[1]*aa.shape[2],aa.shape[3]))
        aa=lasagne.nonlinearities.softmax(aa)
        aa=T.reshape(aa,nn)
        aa=T.transpose(aa,(0,3,1,2))
        return(aa)



    def get_output_shape_for(self, input_shape):
        return (input_shape)

def extra_pars(network,l):
    if ('pad' in l):
            network.pad=l['pad']
    if ('stride' in l):
            network.stride=l['stride']
    return (network)

def get_nonlinearity(l,tinout):

    s1=l['non_linearity']
    if ('rectify' in s1):
        f=lasagne.nonlinearities.rectify
    elif ('rect_sym' in s1):
        f=make_net.rect_sym
    elif ('sigmoid' in s1):
        f=lasagne.nonlinearities.sigmoid
    elif ('tanh' in s1):
        scale_in=tinout[0]
        scale_out=tinout[1]
        if ('tinout' in l):
            scale_in=l['tinout'][0]
            scale_out=l['tinout'][1]
        f=lasagne.nonlinearities.ScaledTanH(scale_in=scale_in,scale_out=scale_out)
    elif ('softmax' in s1):
        f=lasagne.nonlinearities.softmax
    else:
        f=lasagne.nonlinearities.linear
    return(f)

def build_cnn_on_pars(input_var, PARS, input_layer=None, num_class=None):

    add_on=(input_layer is not None)
    network={}
    input_la=None
    if (input_layer is not None):
        input_la=input_layer
    for l in PARS['layers']:
        upd=True
        gain=1.
        if ('gain' in l):
            gain=l['gain']
        prob=(.5,.5)
        if ('global_prob' in PARS):
            prob=PARS['global_prob']
        tinout=(1.,1.)
        if ('global_tinout' in PARS):
            tinout=PARS['global_tinout']
        if ('prob' in l):
            prob=l['prob']
        if ('force_global_prob' in PARS and 'prob' in l and l['prob'][1]!=-1):
            prob=PARS['force_global_prob']
        prob=tuple([np.float32(i) for i in prob])
        #prob.shape=(1,2)
        nonlin=lasagne.nonlinearities.identity
        if 'non_linearity' in l:
            nonlin=get_nonlinearity(l,tinout)
        if ('parent' in l):
            ip=l['parent']
            if (type(ip)==list):
                if (not 'concat' in l['name']):
                    for i,lay in enumerate(input_la):
                        layer_list=[network[ipl][i] for ipl in ip]
                        input_la[i]=lasagne.layers.ElemwiseMergeLayer(layer_list,T.maximum,name='merge')
                else:
                    for i,lay in enumerate(input_la):
                        layer_list=[network[ipl][i] for ipl in ip]
                        input_la[i]=lasagne.layers.ElemwiseMergeLayer(layer_list,T.maximum,name='merge')
            elif (not add_on):
                input_la=network[ip]
        if (type(input_la) is not list):
            input_la=[input_la,]
        layer_list=[]
        if 'input' in l['name']:
            if input_var.ndim==5:
                layer_list=[]
                for j in range(l['augment']):
                    name=None
                    if (len(layer_list)==0):
                            name=l['name']
                    layer_list.append(lasagne.layers.InputLayer(shape=(None, l['num_input_channels'], l['dimx'], l['dimy']),name=name,
                                        input_var=input_var[j]))
            else:
                layer_list.append(lasagne.layers.InputLayer(shape=(None, l['num_input_channels'], l['dimx'], l['dimy']),name=l['name'],
                                        input_var=input_var))
        elif 'feat' in l['name']:
            for lay in input_la:
                if (len(layer_list)==0):
                            name=l['name']
                layer_list.append(lasagne.layers.FeaturePoolLayer(lay,pool_size=l['pool_size'],name=name))
        elif 'drop' in l['name']:
                for lay in input_la:
                        name=None
                        # Only in first step
                        if (len(layer_list)==0):
                            name=l['name']
                        # Once means create the random filter on inputs only once and use it for all shared layers.
                        if (len(layer_list)==0 or 'once' not in l):
                            blay=BnoiseLayer(lay,p=l['drop'],name='noise')
                        layo=lasagne.layers.ElemwiseMergeLayer([lay,blay],merge_function=T.mul,name=name)
                        #layo=lasagne.layers.DropoutLayer(lay,p=l['drop'],name=name)
                        layer_list.append(layo)
        elif 'conv' in l['name']:
                if ('full' in l):
                    if (PARS['train']):
                        filter_size=input_la[0].output_shape[2:]
                        PARS['dense_filter_size']=filter_size
                    else:
                        filter_size=PARS['dense_filter_size']
                else:
                    filter_size=l['filter_size']

                for lay in input_la:
                    if (len(layer_list)==0):
                        if ('R' not in l['name']):
                            convp=lasagne.layers.Conv2DLayer(lay, num_filters=l['num_filters'], filter_size=filter_size,
                                nonlinearity=nonlin,b=None,
                                W=lasagne.init.GlorotUniform(),name=l['name'])
                        else:
                            convp=Conv2dLayerR.Conv2DLayerR(lay, num_filters=l['num_filters'], filter_size=filter_size,
                                nonlinearity=nonlin,W=lasagne.init.GlorotUniform(gain=gain),
                                R=lasagne.init.GlorotUniform(gain=gain),prob=prob,name=l['name'], b=None)
                    else:
                        if ('R' not in l['name']):
                             convp=lasagne.layers.Conv2DLayer(
                                lay,  num_filters=l['num_filters'], filter_size=filter_size,
                                nonlinearity=nonlin,W=layer_list[0].W, b=layer_list[0].b)
                        else:
                            convp=Conv2dLayerR(
                                lay,  num_filters=l['num_filters'], filter_size=filter_size,
                                nonlinearity=nonlin,W=layer_list[0].W, b=layer_list[0].b)
                    convp=extra_pars(convp,l)
                    layer_list.append(convp)
        elif 'batch' in l['name']:
            for lay in input_la:
                name=None
                if (len(layer_list)==0):
                    name=l['name']
                    gamma=None
                    beta=None
                    if ('gamma' in l):
                        gamma=lasagne.init.Constant(1)
                        beta=lasagne.init.Constant(0)
                    convb=lasagne.layers.BatchNormLayer(lay,name=name,beta=beta, gamma=gamma, mean=lasagne.init.Constant(0), inv_std=lasagne.init.Constant(1))
                else:
                    convb=lasagne.layers.BatchNormLayer(lay,beta=layer_list[0].beta, gamma=layer_list[0].gamma, mean=layer_list[0].mean, inv_std=layer_list[0].inv_std)

                layer_list.append(convb)
        elif 'scale' in l['name']:
            for lay in input_la:
                stand=SclLayer(lay,scale=l['fac'],name=l['name'])
                layer_list.append(stand)
        elif 'pool' in l['name']:
            mode=None
            if ('mode' in l):
                mode=l['mode']
            for lay in input_la:
                name=None
                if (len(layer_list)==0):
                    name=l['name']
                if (mode is None):
                        layp=lasagne.layers.MaxPool2DLayer(lay, name=name,pool_size=l['pool_size'],)
                else:
                    layp=lasagne.layers.Pool2DLayer(lay, name=name,pool_size=l['pool_size'],mode=mode)
                layp=extra_pars(layp,l)
                layer_list.append(layp)
        elif 'non_linearity' in l['name']:
            for lay in input_la:
                name=None
                if (len(layer_list)==0):
                    name=l['name']
                layer_list.append(lasagne.layers.NonlinearityLayer(lay,nonlinearity=nonlin,name=name))
        elif 'dense' in l['name']:
                num_units=l['num_units']
                if ('final' in l and num_class is not None):
                    num_units=num_class
                for lay in input_la:
                    if (len(layer_list)==0):
                        layer_list.append(lasagne.layers.DenseLayer(lay,name=l['name'],num_units=num_units,W=lasagne.init.GlorotUniform(gain=gain),
                                                                    b=None, nonlinearity=nonlin))
                    else:
                        layer_list.append(lasagne.layers.DenseLayer(lay,num_units=num_units,nonlinearity=nonlin,
                                          W=layer_list[0].W, b=layer_list[0].b))
        elif 'newdens' in l['name']:
                num_units=l['num_units']
                if ('final' in l and num_class is not None):
                    num_units=num_class


                for lay in input_la:
                    input_dim=np.prod(lay.output_shape[1:])
                    Rz=np.float32(np.ones((1,1)))
                    R=np.float32(np.ones((1,1)))
                    # Reading in existing network, adjust masks to existing parameter values later on
                    if ('use_existing' in PARS and PARS['use_existing']):
                        Wz=np.float32(np.ones((input_dim,num_units)))
                        if (prob[1]>=0):
                            Rz=np.float32(np.ones((input_dim,num_units)))
                    else:
                        Wz=np.float32(np.random.rand(input_dim,num_units)<prob[0])
                        if (prob[1]>=0):
                            Rz=np.float32(np.random.rand(input_dim,num_units)<prob[0])
                    std=np.sqrt(6./(input_dim+num_units))
                    W=np.float32(np.random.uniform(-std,std,(input_dim,num_units)))*Wz
                    if (Rz.shape[0] > 1):
                        R=np.float32(np.random.uniform(-std,std,(input_dim,num_units)))*Rz
                    if (len(layer_list)==0):
                        layer_list.append(newdense.NewDenseLayer(lay,name=l['name'],num_units=num_units,
                                                                    W=W,#lasagne.init.GlorotUniform(gain=gain),
                                                                    R=R,#lasagne.init.GlorotUniform(gain=gain),
                                                                    Wzero=Wz, Rzero=Rz,b=None, prob=prob,nonlinearity=nonlin))
                    else:
                        layer_list.append(lasagne.layers.DenseLayer(lay,num_units=num_units,nonlinearity=nonlin,
                                          W=layer_list[0].W, R=layer_list[0].R, Wzero=Wz, Rzero=Rz, b=layer_list[0].b))
        elif 'sparse' in l['name']:
            for lay in input_la:

                if (len(layer_list)==0):
                    if ('R' in l['name']):
                        layer_list.append(newdensesparse.SparseDenseLayer(lay,num_units=l['num_units'],
                                                          b=None,nonlinearity=nonlin,name=l['name']))
                    else:
                        layer_list.append(densesparse.SparseDenseLayer(lay,num_units=l['num_units'],
                                                          b=None,nonlinearity=nonlin,name=l['name']))
                else:
                    if ('R' in l['name']):
                        layer_list.append(newdensesparse.SparseDenseLayer(lay,num_units=l['num_units'],
                                                          b=None,nonlinearity=nonlin,name=l['name']))
                    else:
                        layer_list.append(densesparse.SparseDenseLayer(lay,num_units=l['num_units'],
                                        nonlinearity=nonlin,
                                          W=layer_list[0].W, b=layer_list[0].b))
        elif 'reshape' in l['name']:
            for lay in input_la:
                layer_list.append(lasagne.layers.ReshapeLayer(lay,([0],)+l['shape'],name=l['name']))
        elif 'global_average' in l['name']:
            for lay in input_la:
                name=None
                if (len(layer_list)==0):
                        name=l['name']
                layer_list.append(lasagne.layers.Pool2DLayer(lay,name=name,pool_size=lay.output_shape[2:],mode='average_inc_pad'))

        elif 'concat' in l['name']:
            for lay in input_la:
                layer_list.append(lay)
        if upd:
            if ('merge' in l):
                        if l['merge']=='max':
                            network[l['name']]=[lasagne.layers.ElemwiseMergeLayer(layer_list,T.maximum,name='merge_auto'),]
                        else:
                            d=1./len(layer_list)
                            network[l['name']]=[lasagne.layers.ElemwiseSumLayer(layer_list,coeffs=d,name='merge_auto'),]
            else:
                        network[l['name']]=layer_list

            sin=sout=None
            la=network[l['name']][0]
            if (hasattr(la,'input_shape')):
                sin=la.input_shape[1:]
            if (hasattr(la,'output_shape')):
                sout=la.output_shape[1:]
            cc=0
            if (hasattr(la,'input_layer') and la.name is not None):
                    cc=lasagne.layers.count_params(la,trainable=True)-lasagne.layers.count_params(la.input_layer,trainable=True)

            print(l['name'],'length',len(network[l['name']]),'input',sin,'output',sout,'num params',cc)

        if ('final' in l):
            fnet=network[l['name']][0]
        elif ('finals' in l):
            layer_list=[]
            for ll in network[l['name']]:
                layer_list.append(lasagne.layers.dimshuffle(ll,('x',0,1,2,3)))
            fnet=lasagne.layers.ConcatLayer(layer_list,0)

        #  Not in add on regime any more (just for first step
        add_on=False

    if ('net' in PARS):
        if (os.path.isfile(PARS['net']+'.npy') and PARS['use_existing']):
            print('net name',PARS['net']+'.npy')
            spars=np.load(PARS['net']+'.npy')
            spars32=[]
            for p in spars:
                 if (p.dtype == np.float32):
                        pp=p
                 else:
                        pp=np.float32(p)
                 spars32.append(pp)
            lasagne.layers.set_all_param_values(fnet,spars32)
            layers=lasagne.layers.get_all_layers(fnet)
            for l in layers:
                if ('newdens' in l.name):
                    WW=np.array(l.W.eval())
                    RR=np.array(l.R.eval())
                    l.Wzero=np.float32(WW!=0)*np.float32(np.random.rand(WW.shape[0],WW.shape[1])<prob[0])
                    if l.Rzero.shape[0] > 1:
                        l.Rzero=np.float32(RR!=0)*np.float32(np.random.rand(RR.shape[0],RR.shape[1])<prob[0])
                    if ('global_prob' in PARS and PARS['global_prob'][1]==0):
                        l.Rzero=np.float32(np.zeros(np.shape(RR)))

    if ('NOT_TRAINABLE' in PARS or 'REMOVE' in PARS or 'INSERT_LAYERS' in PARS):
        layers=lasagne.layers.get_all_layers(fnet)
        # The layers will not be modified any more
        if ('NOT_TRAINABLE' in PARS):
            for n in PARS['NOT_TRAINABLE']:
                    for l in layers:
                        if l.name==n:
                            if ('conv' in n or 'dens' in n):
                              l.params[l.W].remove('trainable')
                              if (l.b is not None):
                                l.params[l.b].remove('trainable')
                              if ('R' in n):
                                  l.params[l.R].remove('trainable')
                            elif ('batch' in n):
                                l.params[l.beta].remove('trainable')
                                l.params[l.gamma].remove('trainable')
                            elif ('sparse' in n):
                                l.params[l.W].remove('trainable')
                                #l.params[l.b].remove('trainable')
                                if('R' in n):
                                    l.params[l.R].remove('trainable')
                            break
        #These layers will be removed
        if ('REMOVE' in PARS and PARS['REMOVE'] is not None):
            for n in PARS['REMOVE']:
                try:
                    a=int(n)
                    l=layers[a]
                    layers.remove(l)
                except ValueError:
                    for l in layers:
                        if l.name==n:
                            layers.remove(l)
                            break
                if n == PARS['REMOVE'][-1]:
                        if (hasattr(l,'input_layer')):
                            fnet=l.input_layer
                        else:
                            fnet=lasagne.layers.ElemwiseMergeLayer(l.input_layers,T.maximum,name='merge')
        # These layers will be inserted.
        # TODO - make this a list of inserted networks that all merge into a layer in the existing network
        if ('INSERT_LAYERS' in PARS):
            prs={}
            new_layers_pars=PARS['INSERT_LAYERS']
            prs['layers']=new_layers_pars
            inlayer=new_layers_pars[0]['parent']
            for l in layers:
                 if l.name==inlayer:
                    fnet_new=build_cnn_on_pars(input_var,prs,l,num_class=num_class)
                    break
            if ('RECONNECT_MULT' in PARS):
                for l in layers:
                     if l.name==PARS['RECONNECT_MULT']:
                        if (hasattr(l,'input_layer')):
                            llist=[fnet_new,l.input_layer]
                        else:
                            llist=[fnet_new]+l.input_layers
                        #print('max values of Ws',T.max(l.input_layer.W).eval(),T.mean(l.input_layer.W).eval(),T.std(l.input_layer.W).eval())
                        #tm=np.abs(T.mean(l.input_layer.W).eval())
                        #sm=np.float32(T.std(l.input_layer.W).eval()*.2)
                        #fnet_new.W*=sm
                        #fnet_new.input_layer.W*=sm
                        l.input_layer=lasagne.layers.ElemwiseMergeLayer(llist,T.maximum,name='merge')
                        break
            elif ('RECONNECT' in PARS):
                for l in layers:
                     if l.name==PARS['RECONNECT']:
                         l.input_layer=fnet_new
                         break;
            else:
                fnet=fnet_new



    return fnet


def make_file_from_params(network,NETPARS):


    # for pp in NETPARS:
    #     if 'name' not in pp:
    #         if 'dict' not in pp:
    #             s=
    ss=[]
    for key, value in NETPARS.iteritems():
        if (key=='RECONNECT' or key=='RECONNECT_MULT' or key=='REMOVE' or key=='INSERT_LAYERS'):
            continue
        if (key != 'layers'):
                if (type(value) != dict):
                    if(type(value) != list):
                        if (type(value)==tuple and type(value[0])==str):
                            s=key+':'
                            for r in value:
                                s=s+r+","
                        else:
                            s=key+':'+str(value)
                    else:
                        s=key+':'
                        for r in value:
                            if (type(r)==str):
                                s=s+r+","
                            else:
                                s=s+str(r)+','
                else:
                    s='dict:'+key
                    for skey, svalue in value.iteritems():
                        s=s+';'+skey+':'+str(svalue)
                ss.append(s)
    layers=lasagne.layers.get_all_layers(network)
    p=None
    for l in layers:
        if (l.name is not None and type(l) is not SclLayer):
            if ('input' in l.name):
                s='name:'+l.name
            elif ('conv' in l.name):
                if (hasattr(l.nonlinearity,'func_name')):
                    sfunc=l.nonlinearity.func_name
                else:
                    sfunc='tanh'+';tinout:('+str(l.nonlinearity.scale_in)+\
                          ','+str(l.nonlinearity.scale_out)+')'
                s='name:'+l.name+';num_filters:'+str(l.num_filters)+';pad:'+str(l.pad)+';filter_size:'\
                  +str(l.filter_size)+';stride:'+str(l.stride)+';non_linearity:'+sfunc
                if (hasattr(l,'prob')):
                    s=s+';prob:'+str(l.prob)
            elif ('reshape' in l.name):
                s='name:'+l.name+';shape:'+str(l.shape[1:])
            elif ('sparse' in l.name):
                if (hasattr(l.nonlinearity,'func_name')):
                    sfunc=l.nonlinearity.func_name
                else:
                    sfunc='tanh'+';tinout:('+str(l.nonlinearity.scale_in)+\
                          ','+str(l.nonlinearity.scale_out)+')'
                s='name:'+l.name+';num_units:'+str(l.num_units)+';non_linearity:'+sfunc
            elif ('noise' in l.name):
                p=l.p
                s=None
            elif ('scale' in l.name):
                s='name:'+l.name+';fac:'+str(l.fac)
            elif ('non_linearity' in l.name):
                sfunc='lasagne.nonlinearity.'+l.nonlinearity.func_name
                s='name:'+l.name+';non_linearity:'+sfunc
            elif ('batch' in l.name):
                s='name:'+l.name
            elif ('feat' in l.name):
                s='name:'+l.name+';pool_size:'+str(l.pool_size)
            elif ('drop' in l.name):
                # Regular drop layer.
                if (p is None):
                    p=l.p
                s='name:'+l.name+';drop:'+str(p)
            elif ('pool' in l.name):
                if (hasattr(l,'mode')):
                    s='name:'+l.name+';pool_size:'+str(l.pool_size)+';stride:'+str(l.stride)+';pad:'+str(l.pad)+';mode:'+str(l.mode)
                else:
                    s='name:'+l.name+';pool_size:'+str(l.pool_size)+';stride:'+str(l.stride)+';pad:'+str(l.pad)
            elif ('dens' in l.name):
                if (hasattr(l.nonlinearity,'func_name')):
                    sfunc=l.nonlinearity.func_name
                else:
                    sfunc='tanh'+';tinout:('+str(l.nonlinearity.scale_in)+\
                          ','+str(l.nonlinearity.scale_out)+')'
                s='name:'+l.name+';num_units:'+str(l.num_units)+';non_linearity:'+sfunc
                if (hasattr(l,'prob')):
                    s=s+';prob:'+str(l.prob)
            elif ('global_average' in l.name):
                s='name:'+l.name
            elif ('merge' in l.name):
                s=None
                continue
            if (not 'input' in l.name and not 'noise' in l.name):
                if (hasattr(l,'input_layer') and 'merge' not in l.input_layer.name):
                    if (l.input_layer.name is not None):
                        s=s+';parent:'+l.input_layer.name
                elif ('drop' in l.name):
                    s=s+';parent:'+l.input_layers[0].name
                else:
                    lin=l
                    # This is simply a layer merging a bunch of same networks working
                    # on transformed images.
                    if l.input_layer.name=='merge_auto':
                        s=s+';parent:'+l.input_layer.input_layers[0].name
                        # Fix the line for the incoming layer
                        ss[-1]=ss[-1]+';merge:max' # TODO: Should adapt to merge function
                    else:
                        if 'merge' in l.input_layer.name:
                            lin=l.input_layer
                        s=s+';parent:['
                        for i,ll in enumerate(lin.input_layers):
                            if (i==0):
                                s=s+ll.name
                            else:
                                s=s+','+ll.name
                        s=s+']'
            if (s is not None):
                ss.append(s)


    ss[-1]=ss[-1]+';final:final'

    f=open(NETPARS['output_net']+'.txt','w')
    for s in ss:
        f.write(s+'\n')
    f.close()
    PARS={}
    parse_net_pars.parse_text_file(NETPARS['output_net'],PARS)
    return(PARS)

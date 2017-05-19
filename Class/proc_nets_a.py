import numpy as np
import pylab as py
import sys
import manage_OUTPUT
from scipy import linalg
import data
import parse_net_pars
import make_net
import lasagne
import run_compare
import theano.tensor as T
import theano
import run_class

parms={}
parms=manage_OUTPUT.process_args(sys.argv,parms)

PAR=[]
PARA=[]
PARW=None
PARR=None
num_nets=parms['stop']-parms['start']
for j in range(parms['start'],parms['stop']):
    PAR=np.load(parms['net']+'_'+str(j)+'.npy')
    if PARA==[]:
        PARA=PAR
    else:
        for i,p in enumerate(PARA):
            p=p+PAR[i]
    # svw=linalg.svd(PARW/j)
    # svr=linalg.svd(PARR/j)
    # py.figure(2)
    # py.plot(svw[1])
    # py.plot(svr[1])
    # py.show()
PARAC=[]
for p in PARA:
    p=np.floatX(p/num_nets)
    PARAC.append(np.copy(p))

NETPARS={}
input_var =  T.tensor4('inputs')
target_var = T.ivector('target')
parse_net_pars.parse_text_file(parms['net'],NETPARS,lname='layers',dump=True)
X_train, y_train, X_val, y_val, X_test, y_test=data.get_train(NETPARS)
dims=X_train.shape[1:]
NETPARS['layers'][0]['dimx']=dims[1]
NETPARS['layers'][0]['dimy']=dims[2]
NETPARS['layers'][0]['num_input_channels']=dims[0]

network = make_net.build_cnn_on_pars(input_var,NETPARS)


lasagne.layers.set_all_param_values(network,PARA)

def make_func(network,input_var):

    out=lasagne.layers.get_output(network)

    out_func=theano.function([input_var],[out])
    return(out_func)


layers=lasagne.layers.get_all_layers(network)
out_func=make_func(layers[parms['layer_number']],input_var)

out=out_func(X_train[0:100])
batch_size=NETPARS['batch_size']
val_fn,dummy=run_compare.setup_function(network,NETPARS,input_var,target_var,Train=False)
out_test=run_class.iterate_on_batches(val_fn,X_test,y_test,batch_size,typ='Test',agg=True,fac=False,pars=NETPARS)

t=0
for i,l in enumerate(layers):
    if hasattr(l,'R'):
        PARA[t]=PARA[t+1]
    t=t+len(l.params)

lasagne.layers.set_all_param_values(network,PARA)
out_func=make_func(layers[parms['layer_number']],input_var)

outR=out_func(X_train[0:100])
val_fn,dummy=run_compare.setup_function(network,NETPARS,input_var,target_var,Train=False)
out_test=run_class.iterate_on_batches(val_fn,X_test,y_test,batch_size,typ='Test',agg=True,fac=False,pars=NETPARS)
print('done')



def do_svm_COMP(parms,PARA):

    PARW=PARA[parms['layer_number']]
    PARR=PARA[parms['layer_number']+1]
    py.ion()
    svw=linalg.svd(PARW)
    svr=linalg.svd(PARR)
    dx=parms['dimx']
    dy=parms['dimy']
    dz=3
    if (dy>0):
        py.figure(1)
        for j in range(10):
            py.subplot(1,2,1)
            py.imshow(svw[0][:,j].reshape(dz,dx,dy).transpose(1,2,0))
            py.subplot(1,2,2)
            py.imshow(svr[0][:,j].reshape(dz,dx,dy).transpose(1,2,0))
    else:
        py.subplot(1,2,1)
        py.imshow(PARW)
        py.subplot(1,2,2)
        py.imshow(PARR)
    py.figure(2)
    py.plot(svw[1])
    py.plot(svr[1])
    py.show()
    print("done")
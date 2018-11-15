# Run the training
import sys
import numpy as np
import tensorflow as tf
import Conv_net_gpu
import Conv_sparse_aux
import  Conv_net_aux
from Conv_net_aux import run_epoch
from Conv_data import get_data


###
# Script starts here
###

net = sys.argv[1]
gpu_device=None

# Tells you which gpu to use.
if (len(sys.argv)>2):
    print(sys.argv[2])
    gpu_device='/device:GPU:'+sys.argv[2]
print('gpu_device',gpu_device)
print('net', net)

# Process parameters from a text file ( as in _pars )
PARS=Conv_net_aux.process_parameters(net)
train, val, test, dim = get_data(PARS)
if ('re_randomize' in PARS): re_randomize=PARS['re_randomize']
else: re_randomize=None
if ('non_trainable' in PARS): non_trainable=PARS['non_trainable']
else: non_trainable=None


with tf.device(gpu_device):
    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    #config.log_device_placement = True
    with tf.Session(config=config) as sess:
        OPS={}
        OPS['x'] = tf.placeholder(tf.float32, shape=[None, dim, dim, PARS['nchannels']], name="x")
        OPS['y_'] = tf.placeholder(tf.float32, shape=[None, PARS['n_classes']], name="y")
        OPS['Train'] = tf.placeholder(tf.bool, name="Train")
        Conv_net_aux.setup_net(PARS,OPS)

        # Initialize variables
        sess.run(tf.global_variables_initializer())
        #zero_out_weights(PARS,VS,sess)

        # Initial test accuracy
        run_epoch(test,-1,OPS,PARS,sess,type='Test')
        # Run training epochs
        for i in range(PARS['num_epochs']):  # number of epochs
            run_epoch(train,i,OPS,PARS,sess)
            if (np.mod(i, 1) == 0):
                run_epoch(val,i,OPS,PARS,sess,type='Val')
                sys.stdout.flush()

        # For those layers that are converted to sparse, either use current values or if they
        # are in rerandomize - reinitialize them.
        if ('sparse' in PARS):
            WRS, sparse_shape=Conv_net_gpu.get_parameters_s(OPS['VS'],PARS['sparse'],OPS['TS'],re_randomize=re_randomize)
            WR=Conv_net_gpu.get_parameters(OPS['VS'],PARS, re_randomize=re_randomize)

        # Final test accuracy
        else:
            Conv_net_aux.finalize(test,OPS,PARS,net,sess)


    # Recreate graph with existing value of parameters
    if ('sparse' in PARS):
      Conv_net_aux.sparse_process_parameters(PARS)

      SDS = None
      tf.reset_default_graph()
      TS=[]
      VS=[]

      with tf.Session(config=config) as sess:

        SP=Conv_sparse_aux.convert_conv_layers_to_sparse(sparse_shape,WRS,sess,PARS)
        OPS={}
        OPS['x'] = tf.placeholder(tf.float32, shape=[PARS['batch_size'], dim, dim, PARS['nchannels']], name="x")
        OPS['y_'] = tf.placeholder(tf.float32, shape=[PARS['batch_size'], PARS['n_classes']], name="y")
        OPS['Train'] = tf.placeholder(tf.bool, name="Train")

        Conv_net_aux.setup_net(PARS, OPS,WR=WR,SP=SP,non_trainable=non_trainable)

        SS=Conv_sparse_aux.get_sparse_parameters(OPS['VS'])

        # Initialize variables
        sess.run(tf.global_variables_initializer())
        Conv_net_gpu.zero_out_weights(PARS,VS,sess)

        # Run on test set before starting to train
        run_epoch(test,-1,OPS,PARS,sess,type='Test')

        # Get some stats on weights.
        SDS=Conv_sparse_aux.get_weight_stats(SS,update=True)
        if (not 'SDS' in PARS or not PARS['SDS']):
            SDS=None

        for i in range(PARS['num_epochs_sparse']):  # number of epochs
                run_epoch(train,i,OPS,PARS,sess,type='Train_sparse')
                # transpose W or R for sparse layer computed once for each epoch
                # NOT for each batch - still works fine.
                Conv_sparse_aux.F_transpose_and_clip(SS,sess,SDS)
                run_epoch(val,i,OPS,PARS,sess,type='Val')
                sys.stdout.flush()
        Conv_sparse_aux.get_weight_stats(SS)
        for sp in PARS['sparse']:
            Conv_sparse_aux.compare_params_sparse(sp, sparse_shape, OPS['VS'], WR)
        Conv_net_aux.finalize(test,OPS,PARS,net,sess)

print("DONE")
sys.stdout.flush()



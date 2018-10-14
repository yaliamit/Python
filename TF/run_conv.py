# Run the training
import time
import sys
import numpy as np
import tensorflow as tf
import Conv_net_gpu
from Conv_net_gpu import  zero_out_weights,  \
     get_parameters, get_parameters_s
import Conv_sparse_aux
from Conv_net_aux import process_parameters,print_results
from Conv_data import get_data, rotate_dataset_rand


def setup_net(PARS, x, y_, Train, WR=None, SP=None, non_trainable=None):
    # Create the network architecture with the above placeholdes as the inputs.
    # TS is a list of tensors or tensors + a list of associated parameters (pool size etc.)
    loss, accuracy, TS = Conv_net_gpu.recreate_network(PARS, x, y_, Train,WR=WR,SP=SP)
    VS = tf.trainable_variables()
    VS.reverse()

    dW_OPs, lall = Conv_net_gpu.back_prop(loss, accuracy, TS, VS, x, PARS,non_trainable=non_trainable)
    return (loss, accuracy, TS, VS, dW_OPs, lall)

def run_epoch(train,i,type='Train',shift=None):
    t1 = time.time()
    # Randomly shuffle the training data
    ii = np.arange(0, train[0].shape[0], 1)
    if (type=='Train'):
        np.random.shuffle(ii)
    tr = train[0][ii]
    if (shift is not None and type=='Train'):
        tr=rotate_dataset_rand(tr,shift=shift,gr=0)

    y = train[1][ii]
    lo = 0.
    acc = 0.
    ca = 0.
    batch_size=PARS['batch_size']
    for j in np.arange(0, len(y), batch_size):
        batch = (tr[j:j + batch_size], y[j:j + batch_size])
        if (type=='Train'):
            grad = sess.run(dW_OPs, feed_dict={x: batch[0], y_: batch[1], Train: True})
            acc += grad[-2]
            lo += grad[-1]
        else:
            act, lot = sess.run([accuracy, loss], feed_dict={x: batch[0], y_: batch[1], Train: False})
            acc += act
            lo += lot
        ca += 1

    print('Epoch time', time.time() - t1)
    print_results(type, i, lo/ca, acc/ca)
    return acc / ca, lo / ca

def read_model(net):
    saver = tf.train.import_meta_graph('tmp/model_' + net + '.meta')
    saver.restore(sess, 'tmp/model_' + net)
    graph = tf.get_default_graph()
    return(graph)

###
# Script starts here
###

net = sys.argv[1]
gpu_device=None
if (len(sys.argv)>2):
    print(sys.argv[2])
    gpu_device='/device:GPU:'+sys.argv[2]
print('gpu_device',gpu_device)
print('net', net)


PARS=process_parameters(net)
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
        x = tf.placeholder(tf.float32, shape=[None, dim, dim, PARS['nchannels']], name="x")
        y_ = tf.placeholder(tf.float32, shape=[None, PARS['n_classes']], name="y")
        Train = tf.placeholder(tf.bool, name="Train")
        loss, accuracy, TS, VS, dW_OPs, lall=setup_net(PARS,x,y_,Train)

        # Initialize variables
        sess.run(tf.global_variables_initializer())
        #zero_out_weights(PARS,VS,sess)

        # Initial test accuracy
        run_epoch(test,-1,type='Test')
        # Run training epochs
        for i in range(PARS['num_epochs']):  # number of epochs
            run_epoch(train,i)
            if (np.mod(i, 1) == 0):
                run_epoch(val,i,type='Val')
                sys.stdout.flush()

        # For those layers that are converted to sparse, either use current values or if they
        # are in rerandomize - reinitialize them.
        if ('sparse' in PARS):
            WRS, sparse_shape=get_parameters_s(VS,PARS['sparse'],TS,re_randomize=re_randomize)
            WR=get_parameters(VS,PARS)

        # Final test accuracy
        else:
            ac, lo= run_epoch(test,i,type='Test')
            print('step,','0,', 'aggegate accuracy,', ac)

    # Recreate graph with existing value of parameters
    if ('sparse' in PARS):
      PARS['step_size']=PARS['eta_sparse_init']
      if ('sparse_batch_size' in PARS):
          PARS['batch_size']=PARS['sparse_batch_size']
      if ('sparse_global_prob' in PARS):
          PARS['force_global_prob']=PARS['sparse_global_prob']
      Rstep_size = list(PARS['force_global_prob'])[1] * PARS['step_size']
      print('Rstep_size', Rstep_size)
      PARS['Rstep_size'] = Rstep_size
      shift = None
      if ('shift' in PARS):
          shift = PARS['shift']
      SDS = None
      tf.reset_default_graph()
      TS=[]
      VS=[]

      with tf.Session(config=config) as sess:

        SP=Conv_sparse_aux.convert_conv_layers_to_sparse(sparse_shape,WRS,sess,PARS)

        x = tf.placeholder(tf.float32, shape=[PARS['batch_size'], dim, dim, PARS['nchannels']], name="x")
        y_ = tf.placeholder(tf.float32, shape=[PARS['batch_size'], PARS['n_classes']], name="y")
        Train = tf.placeholder(tf.bool, name="Train")

        loss, accuracy, TS, VS, dW_OPs, lall = setup_net(PARS, x, y_, Train,WR=WR,SP=SP,non_trainable=non_trainable)

        SS=Conv_sparse_aux.get_sparse_parameters(VS)

        # Initialize variables
        sess.run(tf.global_variables_initializer())
        zero_out_weights(PARS,VS,sess)

        # Run on test set before starting to train
        run_epoch(test,-1,type='Test')


        SDS=Conv_sparse_aux.get_weight_stats(SS,update=True)
        if (not 'SDS' in PARS or not PARS['SDS']):
            SDS=None

        for i in range(PARS['num_epochs_sparse']):  # number of epochs
                run_epoch(train,i,type='Train',shift=shift)
                # transpose W or R for sparse layer computed once for each epoch
                # NOT for each batch - still works fine.
                Conv_sparse_aux.F_transpose_and_clip(SS,sess,SDS)
                run_epoch(val,i,type='Val')
                sys.stdout.flush()
        Conv_sparse_aux.get_weight_stats(SS)
        for sp in PARS['sparse']:
            Conv_sparse_aux.compare_params_sparse(sp, sparse_shape, VS, WR)
        ac, lo= run_epoch(test,i,type='Test')
        print('step,','0,', 'aggegate accuracy,', ac)
        print('sparse comparison after training')
        saver = tf.train.Saver()
        save_path = saver.save(sess, "tmp/model_" + net.split('/')[1])
        print("Model saved in path: %s" % save_path)
        print("DONE")
print("DONE")
sys.stdout.flush()
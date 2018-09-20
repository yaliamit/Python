# Run the training
import time
import sys
import numpy as np
import tensorflow as tf
from Conv_net_gpu import create_network, back_prop, zero_out_weights, find_ts, \
    recreate_network, get_parameters, get_parameters_s
from Conv_net_aux import process_parameters,print_results
from Conv_data import get_data
from Conv_sparse_aux import F_transpose_and_clip, compare_params_sparse, convert_conv_to_sparse, get_weight_stats


def run_epoch(train,i,type='Train'):
    t1 = time.time()
    # Randomly shuffle the training data
    ii = np.arange(0, train[0].shape[0], 1)
    if ('Train'):
        np.random.shuffle(ii)
    tr = train[0][ii]
    y = train[1][ii]
    lo = 0.
    acc = 0.
    ca = 0.
    batch_size=PARS['batch_size']
    for j in np.arange(0, len(y), batch_size):
        batch = (tr[j:j + batch_size], y[j:j + batch_size])
        if (type=='Train'):
            grad = sess.run(dW_OPs, feed_dict={x: batch[0], y_: batch[1], Train: True})
            if (PARS['debug']):
                for j in np.arange(-3, -3 - lall - 1, -1):
                    print(dW_OPs[j].name, 'gradient sd', grad[j].shape, np.std(grad[j]), np.mean(grad[j] == 0))
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


with tf.device(gpu_device):
    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    #config.log_device_placement = True
    with tf.Session(config=config) as sess:
        x = tf.placeholder(tf.float32, shape=[None, dim, dim, PARS['nchannels']], name="x")
        y_ = tf.placeholder(tf.float32, shape=[None, PARS['n_classes']], name="y")
        Train = tf.placeholder(tf.bool, name="Train")


        # Create the network architecture with the above placeholdes as the inputs.
        # TS is a list of tensors or tensors + a list of associated parameters (pool size etc.)
        loss, accuracy, TS = create_network(PARS,x,y_,Train)

        VS = tf.trainable_variables()
        VS.reverse()

        dW_OPs, lall = back_prop(loss,accuracy,TS,VS,x,PARS)

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
        sparse_shape={}
        if ('sparse' in PARS):
            WRS=get_parameters_s(VS,PARS['sparse'])
            WR=get_parameters(VS,PARS)
            for sp in PARS['sparse']:
                sparse_shape[sp]=find_ts(sp,TS).get_shape().as_list()[1:3]
        # Final test accuracy
        else:
            ac, lo= run_epoch(test,i,type='Test')
            print('step,','0,', 'aggegate accuracy,', ac)

        # saver = tf.train.Saver()
        # model_name='model'
        # save_path = saver.save(sess, "tmp/" + model_name)
        # print("Model saved in path: %s" % save_path)



    # Recreate graph with existing value of parameters
    if ('sparse' in PARS):
      PARS['step_size']=PARS['eta_sparse_init']
      Rstep_size = list(PARS['force_global_prob'])[1] * PARS['step_size']
      print('Rstep_size', Rstep_size)
      PARS['Rstep_size'] = Rstep_size
      tf.reset_default_graph()
      TS=[]
      VS=[]

      with tf.Session(config=config) as sess:
        x = tf.placeholder(tf.float32, shape=[PARS['batch_size'], dim, dim, PARS['nchannels']], name="x")
        y_ = tf.placeholder(tf.float32, shape=[PARS['batch_size'], PARS['n_classes']], name="y")
        Train = tf.placeholder(tf.bool, name="Train")

        SP={}
        for sp in PARS['sparse']:
            SP[sp]=convert_conv_to_sparse(sparse_shape[sp],WRS[sp],sess,PARS['force_global_prob'][0])
        loss,accuracy,TS = recreate_network(PARS,x,y_,Train,WR,SP)
        VS = tf.trainable_variables()
        VS.reverse()


        # Get indices of sparse layers:
        SS=[]
        for v in VS:
            if ('sparse' in v.name):
                SS.append(v)
        dW_OPs, lall = back_prop(loss,accuracy,TS,VS,x,PARS)

        # Initialize variables
        sess.run(tf.global_variables_initializer())
        zero_out_weights(PARS,VS,sess)
        run_epoch(test,-1,type='Test')
        SDS=None
        get_weight_stats(SS)
        #print('sparse comparison before training')
        for sp in PARS['sparse']:
           WW=compare_params_sparse(sp,sparse_shape,VS,WR)
        for i in range(PARS['num_epochs_sparse']):  # number of epochs
                run_epoch(train,i)
                # transpose W or R for sparse layer computed once for each epoch
                # NOT for each batch - still works fine.
                F_transpose_and_clip(SS,sess,SDS)
                run_epoch(val,i,type='Val')
                get_weight_stats(SS)
                sys.stdout.flush()
        ac, lo= run_epoch(test,i,type='Test')
        print('step,','0,', 'aggegate accuracy,', ac)
        print('sparse comparison after training')

        for sp in PARS['sparse']:
            WW = compare_params_sparse(sp, sparse_shape, VS, WR)

print("DONE")
sys.stdout.flush()
# Run the training
import parse_net_pars as pp
import time
import sys
import numpy as np
import tensorflow as tf
from Conv_net_gpu import get_data, create_network, back_prop
from Conv_net_aux import process_parameters,zero_out_weights,print_results


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

tf.reset_default_graph()
with tf.device(gpu_device):

    x = tf.placeholder(tf.float32, shape=[None, dim, dim, PARS['nchannels']], name="x")
    y_ = tf.placeholder(tf.float32, shape=[None, PARS['n_classes']], name="y")
    Train = tf.placeholder(tf.bool, name="Train")


    # Create the network architecture with the above placeholdes as the inputs.
    # TS is a list of tensors or tensors + a list of associated parameters (pool size etc.)
    loss, accuracy, TS = create_network(PARS,x,y_,Train)

    VS = tf.trainable_variables()
    VS.reverse()
    dW_OPs, lall = back_prop(loss,accuracy,TS,VS,x,PARS)

    with tf.Session() as sess:
        # Initialize variables
        sess.run(tf.global_variables_initializer())
        for i,v in enumerate(VS):
            print(v.name, v.get_shape().as_list(), np.std(v.eval()))
            # After reversal, i=0 - first trainable variable is last dense layer W.
            #                 i=1 - second trainable variable is last dense layer R
            # Don't zero out these because with large numbers of classes the hinge loss doesn't work.
            if (i > 1):
                zero_out_weights(PARS,v,sess)

        # Differences between W and R
        for t in np.arange(0, len(VS), 2):
            print('t', t, 'zeros', np.sum(VS[t].eval() == 0), np.max(np.abs(VS[t].eval() - VS[t + 1].eval())))

        # Run epochs
        run_epoch(test,-1,type='Test')

        for i in range(PARS['num_epochs']):  # number of epochs
            # ac,lo=\
            run_epoch(train,i)
            if (np.mod(i, 1) == 0):
                run_epoch(val,i,type='Val')
                sys.stdout.flush()

        ac, lo= run_epoch(test,i,type='Test')

        print('step,','0,', 'aggegate accuracy,', ac)

        saver = tf.train.Saver()
        model_name='model'
        save_path = saver.save(sess, "tmp/" + model_name)
        print("Model saved in path: %s" % save_path)
        print("DONE")
        sys.stdout.flush()

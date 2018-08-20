# Run the training
import parse_net_pars as pp
import time
import sys
import numpy as np
import tensorflow as tf
from Conv_net_gpu import get_data, create_network, back_prop
from keras import backend as K


def zero_out_weights(PARS):
    if (PARS['force_global_prob'][1] >= 0 and PARS['force_global_prob'][0] < 1.):
        print('Zeroing out weights at rate ', PARS['force_global_prob'][0])
        shape = v.get_shape().as_list()
        Z = tf.zeros(shape)
        U = tf.random_uniform(shape)
        zero_op = tf.assign(v, K.tf.where(tf.less(U, tf.constant(PARS['force_global_prob'][0])), v, Z))
        sess.run(zero_op)

# Run the iterations of one epoch
def run_epoch(train):
    t1 = time.time()
    # Randomly shuffle the training data
    ii = np.arange(0, train[0].shape[0], 1)
    np.random.shuffle(ii)
    tr = train[0][ii]
    y = train[1][ii]
    lo = 0.
    acc = 0.
    ca = 0.

    for j in np.arange(0, len(y), batch_size):
        batch = (tr[j:j + batch_size], y[j:j + batch_size])
        grad = sess.run(dW_OPs, feed_dict={x: batch[0], y_: batch[1], Train: True})
        if (debug):
            for j in np.arange(-3, -3 - lall - 1, -1):
                print(dW_OPs[j].name, 'gradient sd', grad[j].shape, np.std(grad[j]), np.mean(grad[j] == 0))

        acc += grad[-2]
        lo += grad[-1]
        ca += 1

    print('Epoch time', time.time() - t1)
    return acc / ca, lo / ca


def run_epoch_test(test):
    t1 = time.time()
    # Randomly shuffle the training data

    tr = test[0]
    y = test[1]
    lo = 0.
    acc = 0.
    ca = 0.
    for j in np.arange(0, len(y), batch_size):
        batch = (tr[j:j + batch_size], y[j:j + batch_size])
        act, lot = sess.run([accuracy, loss], feed_dict={x: batch[0], y_: batch[1], Train: False})
        acc += act
        lo += lot
        ca += 1
    print('Epoch time', time.time() - t1)
    return acc / ca, lo / ca


def get_gpu_number():
    import subprocess as commands
    bt=commands.check_output('ssh amit@marx.uchicago.edu "ssh amit@bernie.uchicago.edu \'nvidia-smi -q --id=0 | grep Performance | cut -d: -f2 | cut -dP -f2\' " ', shell=True)
    if (np.int32(bt)<=5):
        return 0
    bt=np.fromstring(commands.check_output('ssh amit@marx.uchicago.edu "ssh amit@bernie \'nvidia-smi -q --id=1 | grep Performance | cut -d: -f2 | cut -dP -f2\' " '))
    if (np.int32(bt) <= 5):
        return 1

    return None


gpu_no=get_gpu_number()
gpu_device='/device:GPU:'+str(gpu_no)
PARS = {}

net = sys.argv[1]  # 'fncrc_try' #'fncrc_deep_tryR_avg'
print('net', net)
pp.parse_text_file(net, PARS, lname='layers', dump=True)
batch_size = PARS['batch_size']
step_size = PARS['eta_init']
num_epochs = PARS['num_epochs']
num_train = PARS['num_train']
data_set = PARS['data_set']
Rstep_size = list(PARS['force_global_prob'])[1] * step_size
print('Rstep_size', Rstep_size)
PARS['Rstep_size']=Rstep_size
PARS['nonlin_scale'] = .5
PARS['avoid_name']='Equal'
model_name = "model"

train, val, test = get_data(data_set=data_set)
num_train = np.minimum(num_train, train[0].shape[0])
train = (train[0][0:num_train], train[1][0:num_train])
dim = train[0].shape[1]
PARS['nchannels'] = train[0].shape[3]
PARS['n_classes'] = train[1].shape[1]
print('n_classes', PARS['n_classes'], 'dim', dim, 'nchannels', PARS['nchannels'])

tf.reset_default_graph()
with tf.device(gpu_device):

    x = tf.placeholder(tf.float32, shape=[None, dim, dim, PARS['nchannels']], name="x")
    y_ = tf.placeholder(tf.float32, shape=[None, PARS['n_classes']], name="y")
    Train = tf.placeholder(tf.bool, name="Train")
    debug = PARS['debug']

    # Create the network architecture with the above placeholdes as the inputs.
    loss, accuracy, TS, sibs = create_network(PARS,x,y_,Train)
    PARS['sibs']=sibs
    TS.reverse()
    for t in TS:
        print(t)
    VS = tf.trainable_variables()
    VS.reverse()
    dW_OPs, lall = back_prop(loss,accuracy,TS,VS,x,PARS)

with tf.Session() as sess:
    # Initialize variables
    sess.run(tf.global_variables_initializer())
    for v in VS:
        print(v.name, v.get_shape().as_list(), np.std(v.eval()))
        zero_out_weights(PARS)

    # Differences between W and R
    for t in np.arange(0, len(VS), 2):
        print('t', t, 'zeros', np.sum(VS[t].eval() == 0), np.max(np.abs(VS[t].eval() - VS[t + 1].eval())))

    # Run epochs
    AC = []
    VAC = []
    ac, lo = run_epoch_test(test)
    print("Final results: before training")
    print("Test loss:\t\t\t{:.6f}".format(lo))
    print("Test acc:\t\t\t{:.6f}".format(ac))
    for i in range(num_epochs):  # number of epochs
        # ac,lo=\
        ac, lo = run_epoch(train)
        if (np.mod(i, 1) == 0):
            # lo,ac = get_stats(train[0][0:num_train],train[1][0:num_train],TS[0])
            # ac, lo = run_epoch_test(train)
            AC.append(ac)
            print("Final results: epoch", i)
            print("Train loss:\t\t\t{:.6f}".format(lo))
            print("Train acc:\t\t\t{:.6f}".format(ac))
            # vlo,vac = get_stats(val[0],val[1],TS[0])
            vac, vlo = run_epoch_test(val)
            VAC.append(vac)
            print("Final results: epoch", i)
            print("Val loss:\t\t\t{:.6f}".format(vlo))
            print("Val acc:\t\t\t{:.6f}".format(vac))
            # print('EPoch',i,'Validation loss, accuracy',vlo,vac)
            sys.stdout.flush()

    AC = np.array(AC)
    VAC = np.array(VAC)
    ac, lo = run_epoch_test(test)
    print("Final results: epoch", i)
    print("Test loss:\t\t\t{:.6f}".format(lo))
    print("Test acc:\t\t\t{:.6f}".format(ac))

    print('step,','0,', 'aggegate accuracy,', ac)
    # plt.plot(AC)
    # plt.plot(VAC)
    # plt.show()
    ACC = np.concatenate([np.expand_dims(AC, axis=1), np.expand_dims(VAC, axis=1)], axis=1)
    np.save('ACC', ACC)
    # Save model
    # tf.add_to_collection("optimizer", train_step)
    saver = tf.train.Saver()
    save_path = saver.save(sess, "tmp/" + model_name)
    print("Model saved in path: %s" % save_path)
    print("DONE")
    sys.stdout.flush()

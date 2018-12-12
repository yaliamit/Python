import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
import sys
from generate_images import generate_image, generate_image_from_estimate
from Conv_net_aux import plot_OUTPUT, process_parameters
from network import recreate_network, conv_layer, fully_connected_layer



def run_epoch(train, i, type='Training',mode='blob'):
        t1 = time.time()
        # Randomly shuffle the training data
        ii=np.arange(0,train[0].shape[0],1)
        if (type=='Training'):
            np.random.shuffle(ii)
        tr = train[0][ii]
        y = train[1][ii]
        cso=0
        acco=0
        disto=0
        ca=0
        HY=[]
        # Run disjoint batches on shuffled data
        for j in np.arange(0, len(y), batch_size):
            batch = (tr[j:j + batch_size], y[j:j + batch_size])
            if (mode=='blob'):
                if (type=='Training'):
                    csi,acc,_=sess.run([cs, accuracy, train_step], feed_dict={x_: batch[0], y_: batch[1], lr_: step_size,
                                           training_:True})
                    acco+=acc[0]
                    disto+=acc[1]
                    cso+=csi
                else:
                    csi, acc, ts = sess.run([cs, accuracy,TS[-1]], feed_dict={x_: batch[0], y_: batch[1], lr_: step_size,
                                                                  training_: False})
                    acco+=acc[0]
                    disto+=acc[1]
                    cso+=csi
                    if (type=='Test'):
                        HY.append(ts)
            elif(mode=='Class'):
                if (type=='Training'):
                    csi,acc,_=sess.run([cs, accuracy, train_step], feed_dict={x_: batch[0], y_: batch[1], lr_: step_size,
                                           training_:True})
                    acco+=acc
                    cso+=csi
                else:
                    csi, acc, ts = sess.run([cs, accuracy], feed_dict={x_: batch[0], y_: batch[1], lr_: step_size,
                                             training_: False})
                    acco+=acc
                    cso += csi

            ca+=1
        print('Epoch time', time.time() - t1)

        print("Final results: epoch", str(i))
        if (mode=='blob'):
            print(type + " dist:\t\t\t{:.6f}".format(disto / ca))
        print(type + " acc:\t\t\t{:.6f}".format(acco / ca))
        print(type + " loss:\t\t\t{:.6f}".format(cso/ca))
        return(HY)


def make_data(num,PARS):
    G=[]
    GC=[]
    num_blobs = np.int32(np.floor(np.random.rand(num) * max_num_blobs) + 1)

    for nb in num_blobs:
        g,gc=generate_image(PARS,num_blobs=nb)
        G.append(np.float32(g))
        GC.append(np.float32(gc))

    return([np.array(G),np.array(GC)])

PARS={}

PARS=process_parameters('_pars/blob1')

cdim=PARS['image_dim']/PARS['coarse_disp']
nchannels=1
minimizer="Adam"

num_epochs=PARS['num_epochs']
batch_size=PARS['batch_size']
image_dim=PARS['image_dim']

max_num_blobs=PARS['max_num_blobs']
step_size=PARS['step_size']

train=make_data(PARS['num_train'],PARS)
val=make_data(PARS['num_val'],PARS)
test=make_data(PARS['num_test'],PARS)

tf.reset_default_graph()

x_ = tf.placeholder(tf.float32, [batch_size, image_dim, image_dim, nchannels])
y_ = tf.placeholder(tf.float32, shape=[batch_size,cdim,cdim,3],name="y")
lr_ = tf.placeholder(tf.float32, shape=[],name="learning_rate")
training_ = tf.placeholder(tf.bool, name="Train")


with tf.Session() as sess:

    cs,accuracy,TS=recreate_network(PARS,x_,y_,training_)
    if (minimizer == "Adam"):
        train_step = tf.train.AdamOptimizer(learning_rate=lr_).minimize(cs)
    elif (minimizer == "SGD"):
        train_step = tf.train.GradientDescentOptimizer(learning_rate=lr_).minimize(cs)
    sess.run(tf.global_variables_initializer())
    for i in range(num_epochs):  # number of epochs
        run_epoch(train,i)
        if (np.mod(i, 1) == 0):
            run_epoch(val, i, type='Val')
            sys.stdout.flush()


    run_epoch(train,0,type='Test')

    # for ind in inds:
    #     generate_image_from_estimate(PARS,HYY[ind],train[0][ind])

    HY=run_epoch(test, 0, type='Test')
    HYY = np.concatenate(HY)
    HYS = HYY[:,:,:,2]>0

    inds = [10, 20, 30, 40, 50, 60, 70, 80]
    for ind in inds:
        generate_image_from_estimate(PARS, HYY[ind], test[0][ind])


    print("Hello")
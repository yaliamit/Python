import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
import sys
import subprocess as commands
from generate_images import show_images, make_data, generate_bigger_images, generate_image_from_estimate, paste_batch
from Conv_net_aux import plot_OUTPUT, process_parameters
from network import recreate_network
from Conv_data import get_data
import pylab as py


def run_epoch(train, PLH,OPS,PARS,sess,i, type='Training'):

        mode='Class'
        if ('blob' in PARS):
            mode='blob'
        t1 = time.time()
        # Randomly shuffle the training data
        batch_size=PARS['batch_size']
        step_size=PARS['step_size']
        ii=np.arange(0,train[0].shape[0],1)
        if (type=='Training'):
            np.random.shuffle(ii)
        tr = train[0][ii]
        y = train[1][ii]
        cso=0
        acco=0
        accon=0
        accop=0
        disto=0
        ca=0
        HY=[]
        thresh=0
        if (type=='Test' and mode=='blob'):
            thresh=PARS['thresh']
        # Run disjoint batches on shuffled data
        for j in np.arange(0, len(y), batch_size):
            batch = (tr[j:j + batch_size], y[j:j + batch_size])
            if (mode=='blob'):
                if (type=='Training'):
                    csi,acc,_=sess.run([OPS['cs'], OPS['accuracy'], OPS['train_step']], 
                                       feed_dict={PLH['x_']: batch[0], PLH['y_']: batch[1], PLH['lr_']: step_size,
                                           PLH['training_']:True, PLH['thresh_']:thresh})
                    accon+=acc[0]
                    accop+=acc[1]
                    disto+=acc[2]
                    cso+=csi
                else:

                    csi, acc, ts = sess.run([OPS['cs'], OPS['accuracy'],OPS['TS']], 
                                            feed_dict={PLH['x_']: batch[0], PLH['y_']: batch[1], PLH['lr_']: step_size,
                                                                  PLH['training_']: False,PLH['thresh_']:thresh})
                    accon+=acc[0]
                    accop+=acc[1]
                    disto+=acc[2]
                    cso+=csi
                    if (type=='Test'):
                        HY.append(ts)

            elif(mode=='Class'):
                if (type=='Training'):
                    csi,acc,ts,_=sess.run([OPS['cs'], OPS['accuracy'], OPS['TS'], OPS['train_step']],
                                       feed_dict={PLH['x_']: batch[0], PLH['y_']: batch[1], PLH['lr_']: step_size,
                                                  PLH['training_']: True})
    
                    acco+=acc
                    cso+=csi
                else:
                    csi, acc,ts = sess.run([OPS['cs'], OPS['accuracy'],OPS['TS']],
                                            feed_dict={PLH['x_']: batch[0], PLH['y_']: batch[1], PLH['lr_']: step_size,
                                                       PLH['training_']: False})
                                            
                    acco+=acc
                    cso += csi

            ca+=1
        print('Epoch time', time.time() - t1)
        print("Final results: epoch", str(i))
        if (mode=='blob'):
            print(type + " dist:\t\t\t{:.6f}".format(disto / ca))
            print(type + " accn:\t\t\t{:.6f}".format(accon / ca))
            print(type + " accp:\t\t\t{:.6f}".format(accop / ca))
            print(type + " loss:\t\t\t{:.6f}".format(cso/ca))
        else:
            print(type + " accn:\t\t\t{:.6f}".format(acco / ca))
            print(type + " loss:\t\t\t{:.6f}".format(cso / ca))
        sys.stdout.flush()
        return(HY)







def run_new(PARS):


    if ('blob' in PARS):
        train=make_data(PARS['num_train'],PARS)
        #show_images(train[0],num=100)
        val=make_data(PARS['num_val'],PARS)
        test=make_data(PARS['num_test'],PARS)
    else:
        mode='Class'
        train, val, test, image_dim = get_data(PARS)
        nchannels = PARS['nchannels']
    
    tf.reset_default_graph()
    PLH={}
    PLH['x_'] = tf.placeholder(tf.float32, [None, image_dim, image_dim, nchannels],name="x_")
    if ('blob' in PARS):
        PLH['y_'] = tf.placeholder(tf.float32, shape=[None,cdim,cdim,num_blob_pars],name="y_")
    else:
        PLH['y_'] = tf.placeholder(tf.float32, shape=[None, PARS['n_classes']], name="y")
    PLH['training_'] = tf.placeholder(tf.bool, name="training_")
    PLH['lr_']=tf.placeholder(tf.float32,name="lr_")
    PLH['thresh_']=tf.placeholder(tf.float32,name="thresh_")
    
    with tf.Session() as sess:
    
        cs,accuracy,TS=recreate_network(PARS,PLH['x_'],PLH['y_'],PLH['training_'],PLH['thresh_'])
        tvars = tf.trainable_variables()
        g_vars = [var for var in tvars if 'newdensf' in var.name]

        if (minimizer == "Adam"):
            train_step = tf.train.AdamOptimizer(learning_rate=PLH['lr_']).minimize(cs,var_list=g_vars)
        elif (minimizer == "SGD"):
            train_step = tf.train.GradientDescentOptimizer(learning_rate=PLH['lr_']).minimize(cs,var_list=g_vars)
        OPS={}
        OPS['cs']=cs; OPS['accuracy']=accuracy; OPS['TS']=TS; OPS['train_step']=train_step



        sess.run(tf.global_variables_initializer())
        #HH=sess.run([TS,],feed_dict={PLH['x_']: train[0][0:500], PLH['y_']: train[1][0:500]})
        for i in range(num_epochs):  # number of epochs
            run_epoch(train,PLH,OPS,PARS,sess,i)
            if (np.mod(i, 1) == 0 and val[0] is not None):
                run_epoch(val, PLH,OPS,PARS,sess,i, type='Val')
                sys.stdout.flush()
    
        print('Running final result of Training')
        run_epoch(train,PLH,OPS,PARS,sess,0,type='Test')
    
        # for ind in inds:
        #     generate_image_from_estimate(PARS,HYY[ind],train[0][ind])
        print('Running final result on Test')
        HY=run_epoch(test,PLH,OPS,PARS,sess, 0, type='Test')
    
        tf.add_to_collection("optimizer", train_step)
        saver = tf.train.Saver()
        commands.check_output('rm -rf '+PARS['model'],shell=True)
        commands.check_output('mkdir ' + PARS['model'], shell=True)
        save_path = saver.save(sess, PARS['model']+'/BL')
        print("Model saved in path: %s" % save_path)
        print("DONE")
        sys.stdout.flush()


def reload(PARS):
    tf.reset_default_graph()

    test, test_batch = generate_bigger_images(PARS)
    show_images(test[0])
    # train = make_data(PARS['num_train'], PARS)
    # val = make_data(PARS['num_val'], PARS)
    # test = make_data(PARS['num_test'], PARS)
    with tf.Session() as sess:
        # Get data
        # Load model info
        saver = tf.train.import_meta_graph(PARS['model'] + '/BL.meta')
        saver.restore(sess,PARS['model']+'/BL')
        graph = tf.get_default_graph()
        # Setup the placeholders from the stored model.
        PLH = {}
        PLH['x_'] = graph.get_tensor_by_name('x_:0')
        PLH['y_'] = graph.get_tensor_by_name('y_:0')
        PLH['lr_'] = graph.get_tensor_by_name('lr_:0')
        PLH['training_'] = graph.get_tensor_by_name('training_:0')
        PLH['thresh_'] = graph.get_tensor_by_name('thresh_:0')

        accuracy=[]

        accuracy.append(graph.get_tensor_by_name('helpers/ACCN:0'))
        accuracy.append(graph.get_tensor_by_name('helpers/ACCP:0'))
        accuracy.append(graph.get_tensor_by_name('helpers/DIST:0'))


        cs = graph.get_tensor_by_name('loss/LOSS:0')
        TS = graph.get_tensor_by_name('LAST:0')
        OPS={}
        OPS['cs']=cs;OPS['accuracy']=accuracy;OPS['TS']=TS
        vars=[]
        for v in tf.trainable_variables():
            vars.append(sess.run(v))
        for i in range(4):
            py.subplot(1,4,i+1)
            py.imshow(vars[0][:,:,0,i])
        py.show()
        # Get the minimization operation from the stored model
        #if (Train):
        #    train_step_new = tf.get_collection("optimizer")[0]
        #test,test_batch=generate_bigger_images(PARS)
        HY = run_epoch(test_batch,PLH,OPS,PARS,sess, 0, type='Test')
        #
        HYY = np.array(HY[0])
        PARS['image_dim']=PARS['big_image_dim']
        HYA=paste_batch(HYY, PARS['old_dim'], PARS['image_dim'],PARS['coarse_disp'],PARS['num_blob_pars'])
        # #HYS = HYY[:,:,:,2]>0
        # #
        #HYA=HYY
        #test=test_batch
        inds = range(len(HYA))
        for ind in inds:
           generate_image_from_estimate(PARS, HYA[ind], test[0][ind],test[1][ind])

net = sys.argv[1]
gpu_device=None


print('net', net)

PARS = {}

PARS = process_parameters(net)

if ('blob' in PARS):
    cdim = PARS['image_dim'] / PARS['coarse_disp']
    nchannels = 1
    minimizer = "Adam"
    image_dim = PARS['image_dim']
    num_blob_pars = PARS['num_blob_pars']
else:
    minimizer=PARS['minimizer']

num_epochs = PARS['num_epochs']
batch_size = PARS['batch_size']


#PARS = process_parameters(net)


if ('train' in net or not 'blob' in PARS):
    run_new(PARS)
else:
    reload(PARS)

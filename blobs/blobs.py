import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
import sys
from generate_images import show_images, make_data, generate_bigger_images, generate_image_from_estimate, paste_batch
from Conv_net_aux import plot_OUTPUT, process_parameters
from network import recreate_network


def run_epoch(train, PLH,OPS,PARS,sess,i, type='Training',mode='blob'):
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
        # Run disjoint batches on shuffled data
        for j in np.arange(0, len(y), batch_size):
            batch = (tr[j:j + batch_size], y[j:j + batch_size])
            if (mode=='blob'):
                if (type=='Training'):
                    csi,acc,_=sess.run([OPS['cs'], OPS['accuracy'], OPS['train_step']], 
                                       feed_dict={PLH['x_']: batch[0], PLH['y_']: batch[1], PLH['lr_']: step_size,
                                           PLH['training_']:True})
                    accon+=acc[0]
                    accop+=acc[1]
                    disto+=acc[2]
                    cso+=csi
                else:
                    csi, acc, ts = sess.run([OPS['cs'], OPS['accuracy'],OPS['TS']], 
                                            feed_dict={PLH['x_']: batch[0], PLH['y_']: batch[1], PLH['lr_']: step_size,
                                                                  PLH['training_']: False})
                    accon+=acc[0]
                    accop+=acc[1]
                    disto+=acc[2]
                    cso+=csi
                    if (type=='Test'):
                        HY.append(ts)

            elif(mode=='Class'):
                if (type=='Training'):
                    csi,acc,_=sess.run([OPS['cs'], OPS['accuracy'], OPS['train_step']],
                                       feed_dict={PLH['x_']: batch[0], PLH['y_']: batch[1], PLH['lr_']: step_size,
                                                  PLH['training_']: True})
    
                    acco+=acc
                    cso+=csi
                else:
                    csi, acc, ts = sess.run([OPS['cs'], OPS['accuracy']],
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
        sys.stdout.flush()
        return(HY)







def run_new(PARS):

    train=make_data(PARS['num_train'],PARS)
    show_images(train[0],num=10)
    val=make_data(PARS['num_val'],PARS)
    test=make_data(PARS['num_test'],PARS)
    
    tf.reset_default_graph()
    PLH={}
    PLH['x_'] = tf.placeholder(tf.float32, [None, image_dim, image_dim, nchannels],name="x_")
    PLH['y_'] = tf.placeholder(tf.float32, shape=[None,cdim,cdim,num_blob_pars],name="y_")
    PLH['training_'] = tf.placeholder(tf.bool, name="training_")
    PLH['lr_']=tf.placeholder(tf.float32,name="lr_")
    
    with tf.Session() as sess:
    
        cs,accuracy,TS=recreate_network(PARS,PLH['x_'],PLH['y_'],PLH['training_'])
        if (minimizer == "Adam"):
            train_step = tf.train.AdamOptimizer(learning_rate=PLH['lr_']).minimize(cs)
        elif (minimizer == "SGD"):
            train_step = tf.train.GradientDescentOptimizer(learning_rate=PLH['lr_']).minimize(cs)
        OPS={}
        OPS['cs']=cs; OPS['accuracy']=accuracy; OPS['TS']=TS; OPS['train_step']=train_step

        sess.run(tf.global_variables_initializer())
        #HH=sess.run([TS,],feed_dict={PLH['x_']: train[0][0:500], PLH['y_']: train[1][0:500]})
        for i in range(num_epochs):  # number of epochs
            run_epoch(train,PLH,OPS,PARS,sess,i)
            if (np.mod(i, 1) == 0):
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
        save_path = saver.save(sess, "_blobs/" + PARS['model'])
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
        saver = tf.train.import_meta_graph('_blobs_clean/' + PARS['model'] + '.meta')
        saver.restore(sess, '_blobs_clean/' + PARS['model'])
        graph = tf.get_default_graph()
        # Setup the placeholders from the stored model.
        PLH = {}
        PLH['x_'] = graph.get_tensor_by_name('x_:0')
        PLH['y_'] = graph.get_tensor_by_name('y_:0')
        PLH['lr_'] = graph.get_tensor_by_name('lr_:0')
        PLH['training_'] = graph.get_tensor_by_name('training_:0')


        accuracy=[]

        accuracy.append(graph.get_tensor_by_name('helpers/ACCN:0'))
        accuracy.append(graph.get_tensor_by_name('helpers/ACCP:0'))
        accuracy.append(graph.get_tensor_by_name('helpers/DIST:0'))


        cs = graph.get_tensor_by_name('loss/LOSS:0')
        TS = graph.get_tensor_by_name('LAST:0')
        OPS={}
        OPS['cs']=cs;OPS['accuracy']=accuracy;OPS['TS']=TS

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

cdim = PARS['image_dim'] / PARS['coarse_disp']
nchannels = 1
minimizer = "Adam"

num_epochs = PARS['num_epochs']
batch_size = PARS['batch_size']
image_dim = PARS['image_dim']

PARS = process_parameters(net)
num_blob_pars = PARS['num_blob_pars']

#run_new(PARS)
reload(PARS)

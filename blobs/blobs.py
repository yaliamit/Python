import tensorflow as tf
import numpy as np
import os
import sys
import subprocess as commands
from generate_images import show_images, acquire_data, generate_bigger_images, generate_image_from_estimate, paste_batch
from Conv_net_aux import process_parameters
from network import recreate_network, run_epoch, get_trainable

import pylab as py

def get_place_holders(PARS,image_dim):


    PLH={}
    if ('corr' not in PARS or not PARS['corr']):
        PLH['x_'] = tf.placeholder(tf.float32, [1, None, image_dim, image_dim, PARS['nchannels']],name="x_")
    else:
        PLH['x_'] = tf.placeholder(tf.float32, [2, None, image_dim, image_dim, PARS['nchannels']], name="x_")
    if ('blob' in PARS):
        cdim = PARS['image_dim'] / PARS['coarse_disp']
        PLH['y_'] = tf.placeholder(tf.float32, shape=[None,cdim,cdim,PARS['num_blob_pars']],name="y_")
    else:
        PLH['y_'] = tf.placeholder(tf.float32, shape=[None, PARS['n_classes']], name="y")
    PLH['training_'] = tf.placeholder(tf.bool, name="training_")
    PLH['lr_']=tf.placeholder(tf.float32,name="lr_")
    PLH['thresh_']=tf.placeholder(tf.float32,name="thresh_")
    PLH['index_']=tf.placeholder(tf.int32,name="index_")
    PLH['global_L2_fac_']=tf.placeholder(tf.float32,name="global_L2_fac_")
    #PLH['use_labels_']=tf.placeholder(tf.bool,name="use_labels_")
    return PLH

def get_place_holders_OPS_from_graph(graph):
        PLH = {}
        PLH['x_'] = graph.get_tensor_by_name('x_:0')
        PLH['y_'] = graph.get_tensor_by_name('y_:0')
        PLH['lr_'] = graph.get_tensor_by_name('lr_:0')
        PLH['training_'] = graph.get_tensor_by_name('training_:0')
        PLH['thresh_'] = graph.get_tensor_by_name('thresh_:0')
        PLH['index_'] = tf.placeholder(tf.int32, name="index_")
        PLH['global_L2_fac_'] = tf.placeholder(tf.float32, name="global_L2_fac_")
        accuracy = []

        accuracy.append(graph.get_tensor_by_name('helpers/ACCN:0'))
        accuracy.append(graph.get_tensor_by_name('helpers/ACCP:0'))
        accuracy.append(graph.get_tensor_by_name('helpers/DIST:0'))

        cs = graph.get_tensor_by_name('loss/LOSS:0')
        TS = graph.get_tensor_by_name('LAST:0')
        OPS = {}
        OPS['cs'] = cs;
        OPS['accuracy'] = accuracy;
        OPS['TS'] = TS

        return PLH, OPS

def get_stats():

    tvars = tf.trainable_variables()
    for t in tvars:
        if 'W' in t.name:
            W=t.eval()
            print(t.name,np.max(W),np.min(W),np.std(W))


def run_new(PARS):

    num_epochs = PARS['num_epochs']
    train, val, test, image_dim=acquire_data(PARS)
    tf.reset_default_graph()
    PLH=get_place_holders(PARS,image_dim)

    with tf.Session() as sess:
    
        cs,accuracy,TS=recreate_network(PARS,PLH)
        g_vars=get_trainable(PARS)
        if (PARS['minimizer'] == "Adam"):
            train_step = tf.train.AdamOptimizer(learning_rate=PLH['lr_']).minimize(cs,var_list=g_vars)
        elif (PARS['minimizer'] == "SGD"):
            train_step = tf.train.GradientDescentOptimizer(learning_rate=PLH['lr_']).minimize(cs,var_list=g_vars)
        OPS={}
        OPS['cs']=cs; OPS['accuracy']=accuracy; OPS['TS']=TS; OPS['train_step']=train_step

        sess.run(tf.global_variables_initializer())

        get_stats()
        for i in range(num_epochs):  # number of epochs
            run_epoch(train,PLH,OPS,PARS,sess,i)
            run_epoch(val, PLH,OPS,PARS,sess,i, type='Val')
            sys.stdout.flush()
    
        print('Running final result of Training')
        run_epoch(train,PLH,OPS,PARS,sess,0,type='Test')
        print('Running final result on Test')
        run_epoch(test,PLH,OPS,PARS,sess, 0, type='Test')
    
        tf.add_to_collection("optimizer", train_step)
        saver = tf.train.Saver()
        commands.check_output('rm -rf '+PARS['model'],shell=True)
        commands.check_output('mkdir ' + PARS['model'], shell=True)
        save_path = saver.save(sess, PARS['model']+'/BL')
        print("Model saved in path: %s" % save_path)
        print("DONE")
        sys.stdout.flush()


def reload(PARS,train=False):

    tf.reset_default_graph()
    if ('blob' in PARS and PARS['blob']):
        np.random.seed(34567)
        test, test_batch = generate_bigger_images(PARS)
    else:
        train, val, test, image_dim = acquire_data(PARS)

    #show_images(test[0])
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
        PLH, OPS = get_place_holders_OPS_from_graph(graph)

        vars=[]
        for v in tf.trainable_variables():
            vars.append(sess.run(v))

        if ('blob' in PARS and PARS['blob']):
            # for i in range(4):
            #     py.subplot(1,4,i+1)
            #     py.imshow(vars[0][:,:,0,i])

            py.show()
            HY = run_epoch(test_batch,PLH,OPS,PARS,sess, 0, type='Test')
            HYY = np.array(HY[0])
            PARS['image_dim']=PARS['big_image_dim']
            HYA=paste_batch(HYY, PARS['old_dim'], PARS['image_dim'],PARS['coarse_disp'],PARS['num_blob_pars'])
            inds = range(len(HYA))
            for ind in inds:
                generate_image_from_estimate(PARS, HYA[ind], test[0][ind],test[1][ind])
        else:
            # Get the minimization operation from the stored model
            if (train):
                train_step_new = tf.get_collection("optimizer")[0]
            for i in range(PARS['num_epochs']):  # number of epochs
                run_epoch(train, PLH, OPS, PARS, sess, i)
                run_epoch(val, PLH, OPS, PARS, sess, i, type='Val')
                sys.stdout.flush()

net = sys.argv[1]
gpu_device=None


print('net', net)

PARS = {}

PARS = process_parameters(net)



if ('train' in net or not os.path.exists(PARS['model'])):
    run_new(PARS)
else:
    reload(PARS)

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
import sys
from generate_images import generate_image, generate_image_from_estimate
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
                    acco+=acc[0]
                    disto+=acc[1]
                    cso+=csi
                else:
                    csi, acc, ts = sess.run([OPS['cs'], OPS['accuracy'],OPS['TS']], 
                                            feed_dict={PLH['x_']: batch[0], PLH['y_']: batch[1], PLH['lr_']: step_size,
                                                                  PLH['training_']: False})
                    acco+=acc[0]
                    disto+=acc[1]
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
        print(type + " acc:\t\t\t{:.6f}".format(acco / ca))
        print(type + " loss:\t\t\t{:.6f}".format(cso/ca))
        sys.stdout.flush()
        return(HY)


def make_data(num,PARS):
    G=[]
    GC=[]
    num_blobs = np.int32(np.floor(np.random.rand(num) * PARS['max_num_blobs']) + 1)

    for nb in num_blobs:
        g,gc=generate_image(PARS,num_blobs=nb)
        G.append(np.float32(g))
        GC.append(np.float32(gc))

    return([np.array(G),np.array(GC)])


def generate_bigger_images(PARS):

        old_dim=np.int32(PARS['image_dim'])
        PARS['old_dim']=old_dim
        PARS['image_dim']=np.int32(PARS['big_image_dim'])
        dim_ratio=PARS['image_dim']/old_dim
        PARS['max_num_blobs']=PARS['max_num_blobs']*old_dim*old_dim
        PARS['num_test']=PARS['big_num_test']
        test = make_data(PARS['num_test'], PARS)
        test_batch=make_batch(test,old_dim,np.int32(PARS['coarse_disp']))
        PARS['batch_size']=np.minimum(PARS['batch_size'],test_batch[0].shape[0])

        return(test_batch)
        #


def reload():
    tf.reset_default_graph()


    train = make_data(PARS['num_train'], PARS)
    val = make_data(PARS['num_val'], PARS)
    test = make_data(PARS['num_test'], PARS)
    with tf.Session() as sess:
        # Get data
        # Load model info
        saver = tf.train.import_meta_graph('_tmp/' + PARS['model'] + '.meta')
        saver.restore(sess, '_tmp/' + PARS['model'])
        graph = tf.get_default_graph()
        # Setup the placeholders from the stored model.
        PLH = {}
        PLH['x_'] = graph.get_tensor_by_name('x_:0')
        PLH['y_'] = graph.get_tensor_by_name('y_:0')
        PLH['lr_'] = graph.get_tensor_by_name('lr_:0')
        PLH['training_'] = graph.get_tensor_by_name('training_:0')


        accuracy=[]
        accuracy.append(graph.get_tensor_by_name('helpers/ACC:0'))
        accuracy.append(graph.get_tensor_by_name('helpers/DIST:0'))
        cs = graph.get_tensor_by_name('loss/LOSS:0')
        TS = graph.get_tensor_by_name('LAST:0')
        OPS={}
        OPS['cs']=cs;OPS['accuracy']=accuracy;OPS['TS']=TS

        # Get the minimization operation from the stored model
        #if (Train):
        #    train_step_new = tf.get_collection("optimizer")[0]
        test=generate_image(PARS)
        HY = run_epoch(test,PLH,OPS,PARS,sess, 0, type='Test')
        #
        HYY = np.array(HY[0])
        PARS['image_dim']=PARS['old_dim']
        inds = range(len(HYY))
        for ind in inds:
             generate_image_from_estimate(PARS, HYY[ind], test[0][ind])
        PARS['image_dim']=PARS['big_image_dim']
        HYA=paste_batch(HYY, PARS['old_dim'], PARS['image_dim'],PARS['coarse_disp'])
        # #HYS = HYY[:,:,:,2]>0
        # #
        inds = range(len(HYA))
        for ind in inds:
           generate_image_from_estimate(PARS, HYA[ind], test[0][ind])


def make_batch(test,old_dim,coarse_disp):

    tbatch=[]
    for t in test[0]:
        for i in np.arange(0,t.shape[0],old_dim):
            for j in np.arange(0,t.shape[1],old_dim):
                tbatch.append(t[i:i+old_dim,j:j+old_dim,:])

    tbatch=np.array(tbatch)
    cbatch=[]
    coarse_dim=np.int32(old_dim/coarse_disp)
    for t in test[1]:
        for i in np.arange(0,t.shape[0],coarse_dim):
            for j in np.arange(0,t.shape[1],coarse_dim):
                cbatch.append(t[i:i+coarse_dim,j:j+coarse_dim,:])
    cbatch=np.array(cbatch)
    batch=[tbatch,cbatch]
    return(batch)


def paste_batch(HYY,old_dim,new_dim,coarse_disp):

    num_per=np.int32(new_dim/old_dim)
    num_per2=num_per*num_per

    cnew_dim=np.int32(new_dim/coarse_disp)
    cold_dim=np.int32(old_dim/coarse_disp)

    HY=[]
    for i in np.arange(0,len(HYY),num_per2):
        hy=np.zeros((cnew_dim,cnew_dim,3))
        nb=np.zeros((new_dim,new_dim,1))
        # j=0
        # for x in np.arange(0,new_dim,old_dim):
        #     for y in np.arange(0,new_dim,old_dim):
        #         nb[x:x+old_dim,y:y+old_dim,0]=batch[0][i+j][:,:,0]
        #         j+=1

        j=0
        for y in np.arange(0,cnew_dim,cold_dim):
            for x in np.arange(0, cnew_dim, cold_dim):
                for k in range(3):
                    hy[x:x+cold_dim,y:y+cold_dim,k]=HYY[i+j][:,:,k]
                j+=1
        HY.append(hy)

    return(HY)




def run_new(PARS):

    train=make_data(PARS['num_train'],PARS)
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
        save_path = saver.save(sess, "_tmp/" + PARS['model'])
        print("Model saved in path: %s" % save_path)
        print("DONE")
        sys.stdout.flush()








net = sys.argv[1]
gpu_device=None

# Tells you which gpu to use.
# if (len(sys.argv)>2):
#     print(sys.argv[2])
#     gpu_device='/device:GPU:'+sys.argv[2]
# print('gpu_device',gpu_device)
print('net', net)

PARS = {}

PARS = process_parameters(net)

cdim = PARS['image_dim'] / PARS['coarse_disp']
nchannels = 1
minimizer = "Adam"

num_epochs = PARS['num_epochs']
batch_size = PARS['batch_size']
image_dim = PARS['image_dim']

PARS = process_parameters('_pars/blob1')
num_blob_pars = PARS['num_blob_pars']

run_new(PARS)
#reload(PARS)

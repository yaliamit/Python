# Run the training
import time
import sys
import numpy as np
import tensorflow as tf
from Conv_net_gpu import create_network, back_prop, zero_out_weights, convert_conv_to_sparse, find_ts, \
    recreate_network, get_parameters, get_parameters_s
from Conv_net_aux import process_parameters,print_results
from Conv_data import get_data


def compare_params_sparse(sp, sh, VS, WR):
    for v in VS:
        if (sp in v.name and 'W' in v.name):
            if ('inds' in v.name):
                minds = v.eval()
            if ('vals' in v.name):
                mvals = v.eval()
            if ('dims' in v.name):
                mdims = v.eval()
    with tf.device("/cpu:0"):
        dm = tf.sparse_to_dense(sparse_indices=minds, sparse_values=mvals, output_shape=mdims)
        DM = dm.eval()
    outfe=WR[sp][0].shape[3]
    infe=WR[sp][0].shape[2]
    fdims=[WR[sp][0].shape[0],WR[sp][0].shape[1]]
    pfdims=np.prod(fdims)
    newshape=[np.int32(DM.shape[0]/outfe),outfe,sh[sp][0],sh[sp][1],infe]
    DM = np.reshape(DM, newshape)
    tt = [[] for i in range(outfe)]
    for i in range(newshape[0]):
        for f in range(newshape[1]):
            ww=np.where(DM[i,f,:,:,0])
            if (len(ww[0])==pfdims):
                tt[f].append(DM[i,f,ww[0],ww[1],0])
    print('sp',sp)
    for f in range(outfe):
        tt[f]=np.array(tt[f])
        print(f,np.max(np.std(tt[f],axis=0)/np.abs(np.mean(tt[f],axis=0))))


# Each layer comes in groups of 9 parameters
def F_transpose_and_clip(VS,SDS=None):

    t=0
    for t in np.arange(0,len(VS),9):
        if (SDS is not None):
                sess.run(tf.assign(VS[t+7],tf.clip_by_value(VS[t+7],-SDS[t+7],SDS[t+7])))
                sess.run(tf.assign(VS[t + 4], tf.clip_by_value(VS[t + 4], -SDS[t + 4], SDS[t + 4])))
        # Indicates there is a real R feedback tensor. Otherwise the length is 1
        if (VS[t+8].get_shape().as_list()[0]==2):
            finds=VS[t+6]
            fvals=VS[t+7]
            fdims=VS[t+8]
        else:
            finds=VS[t+3]
            fvals=VS[t+4]
            fdims=VS[t+5]

        F=tf.SparseTensor(indices=finds,values=fvals,dense_shape=fdims)
        F=tf.sparse_transpose(F)

        sess.run(tf.assign(VS[t+0],F.indices))
        sess.run(tf.assign(VS[t+1],F.values))
        sess.run(tf.assign(VS[t+2],F.dense_shape))


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
        for ss in SS:
            #if ('Wvals' in ss.name):
                V = ss.eval()
                #SDS.append(np.std(V))
                print(ss.name, ss.get_shape().as_list(), np.mean(V),np.std(V))
        #print('sparse comparison before training')
        #for sp in PARS['sparse']:
        #   WW=compare_params_sparse(sp,sparse_shape,VS,WR)
        for i in range(PARS['num_epochs_sparse']):  # number of epochs
                run_epoch(train,i)
                # transpose W or R for sparse layer
                F_transpose_and_clip(SS,SDS)
                if (np.mod(i, 1) == 0):
                    run_epoch(val,i,type='Val')


                sys.stdout.flush()
        ac, lo= run_epoch(test,i,type='Test')
        print('step,','0,', 'aggegate accuracy,', ac)
        print('sparse comparison after training')

        #for sp in PARS['sparse']:
        #    WW = compare_params_sparse(sp, sparse_shape, VS, WR)







print("DONE")
sys.stdout.flush()
import tensorflow as tf
import Conv_net_gpu
import Conv_net_aux
from Conv_net_aux import run_epoch
import Conv_sparse_aux
import sys
import Conv_data
import numpy as np


def get_evaluated_parameters(model_name,gpu_device):
 with tf.Session(gpu_device) as sess:
    saver = tf.train.import_meta_graph('tmp/'+model_name+'.meta')
    saver.restore(sess,'tmp/'+model_name)
    graph = tf.get_default_graph()
    VGS=graph._collections['trainable_variables']

    WR=Conv_net_gpu.get_parameters(VGS,PARS, re_randomize=None)
    SPV=Conv_sparse_aux.get_sparse_parameters_eval(VGS,PARS)
 return WR,SPV


net = sys.argv[1]
additional_epochs=None
if (len(sys.argv)>2):
    additional_epochs=int(sys.argv[2])

gpu_device=None
if (len(sys.argv)>3):
    print(sys.argv[3])
    gpu_device='/device:GPU:'+sys.argv[3]

print('gpu_device',gpu_device)
print('net', net)
print('additional_epochs',additional_epochs)

PARS=Conv_net_aux.process_parameters(net)
train, val, test, dim = Conv_data.get_data(PARS)
if ('re_randomize' in PARS): re_randomize=PARS['re_randomize']
else: re_randomize=None
if ('non_trainable' in PARS): non_trainable=PARS['non_trainable']
else: non_trainable=None
model_name='model_'+net.split('/')[1]
# Get the actual model parameters, then rebuild network with those values

WR,SPV = get_evaluated_parameters(model_name,gpu_device)

# Rebuil network iwth WR and SPV
tf.reset_default_graph()
with tf.Session(gpu_device) as sess:
    OPS={}
    OPS['x'] = tf.placeholder(tf.float32, shape=[PARS['batch_size'], dim, dim, PARS['nchannels']], name="x")
    OPS['y_'] = tf.placeholder(tf.float32, shape=[PARS['batch_size'], PARS['n_classes']], name="y")
    OPS['Train'] = tf.placeholder(tf.bool, name="Train")
    SP = Conv_sparse_aux.convert_vals_to_sparse(SPV)
    Conv_net_aux.setup_net(PARS, OPS,WR=WR, SP=SP)
    sess.run(tf.global_variables_initializer())
    run_epoch(test, -1, OPS,PARS,sess, type='Test')
    if (additional_epochs is not None):
        for i in range(additional_epochs):  # number of epochs
            run_epoch(train,i,OPS,PARS,sess)
            run_epoch(val,i,OPS,PARS,sess,type='Val')
        Conv_net_aux.finalize(test,OPS,PARS,net+'_'+str(additional_epochs),sess)
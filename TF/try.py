import tensorflow as tf
import numpy as np
from Conv_net_aux import process_parameters
from Conv_net_aux import plot_OUTPUT as po
from Conv_data import get_data,rotate_dataset_rand
#po('_OUTPUTS/OUTh_deep_R-br')

net='_pars/fncrc'
PARS=process_parameters(net)
train, val, test, dim = get_data(PARS)

Xtr=rotate_dataset_rand(train[0][0:100],shift=0,flip=False,saturation=True,gr=1)


# print(tf.__version__)
# tf.reset_default_graph()
# infe=2
# outfe=3
# tro=np.float32(np.random.rand(1,4,4,infe))
#
# #print('done')
#
# shape=[3,3,infe,outfe]
# W = tf.convert_to_tensor(np.float32(np.random.rand(shape[0],shape[1],shape[2],shape[3]))) #tf.get_variable('W',shape=shape)
# din=tro.shape[1:]
# dimin=np.prod(din)
# dout=din[0:2]+(shape[3],)
# dimout=np.prod(dout)
#
#
# XX=np.zeros((dimin,)+din)
# t=0
# for i in range(din[0]):
#     for j in range(din[1]):
#         for k in range(din[2]):
#             XX[t,i,j,k]=1
#             t+=1
#
# #x = tf.placeholder(tf.float32, shape=[None, din[0], din[1], din[2]], name="x")
# #conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
#
#
#
#
# with tf.Session() as sess:
#
#     #sess.run(tf.global_variables_initializer())
#     #dpgo=sess.run(dpg)
#     inc=8
#     pin=0
#     indsa=[]
#     valsa=[]
#     for t in range(0,dimin,inc):
#         batch=tf.convert_to_tensor(np.float32(XX[t:t+inc,]))
#         out = sess.run(tf.nn.conv2d(batch,W,strides=[1,1,1,1],padding='SAME'))
#         out=np.reshape(out,(inc,-1))
#         vals=out[out!=0]
#         inds=np.array(np.where(out!=0))
#         inds[0]=inds[0]+t
#         indsa.append(inds.transpose())
#         valsa.append(vals)
#
#     INDS=tf.convert_to_tensor(np.concatenate(indsa,axis=0),dtype=np.int64)
#     VALS=tf.convert_to_tensor(np.concatenate(valsa,axis=0), dtype=np.float32)
#     ndims=tf.convert_to_tensor([dimin,dimout],dtype=np.int64)
#     INDSA=tf.gather(INDS,[1,0],axis=1)
#     ndimsa=tf.gather(ndims,[1,0],axis=0)
#
#     print('tro',tro[:,:,0])
#
#     SP=tf.SparseTensor(indices=INDS,values=VALS,dense_shape=ndims)
#     SP=tf.sparse_transpose(SP)
#     out1=sess.run(tf.nn.conv2d(tro,W,strides=[1,1,1,1],padding='SAME'))
#     ttro=tf.convert_to_tensor(np.reshape(tro,(1,-1)).transpose())
#     convs=tf.sparse_tensor_dense_matmul(SP,ttro)
#     out2=sess.run(convs)
#     out2a=np.reshape(out2,out1.shape)
#     print(out1.shape,out2a.shape)
#     print('out1',out1[0,:,:,1])
#     print('out2a',out2a[0,:,:,1])
# print('done')



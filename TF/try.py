import tensorflow as tf
import h5py
import numpy as np
import pylab as py
import keras
from keras.layers.convolutional import UpSampling2D
# filename = '../_CIFAR100/cifar10_train.hdf5'
# print(filename)
# f = h5py.File(filename, 'r')
# key = list(f.keys())[0]
# # Get the data
# tr = f[key]
# print('tr',tr.shape)
# key = list(f.keys())[1]
# tr_lb=f[key]
# train_data=np.float32(tr[0:45000])/255.

#py.imshow(train_data[0])
#py.show()

im=np.ones((2,4,4,3))
im[0,:,:,1]=np.ones((4,4))*2
im[0,:,:,2]=np.ones((4,4))*3
#im=np.float32(np.reshape(im,(1,4,4,3)))
print(im[0,:,:,1])
im[1]=1.5*im[0]

tf.reset_default_graph()
input=tf.convert_to_tensor(im)

upim=UpSampling2D(size=[2,2])(input)
# aa=np.ones((3,3,3))*3
# aa[:,:,1]=np.ones((3,3))*4
# aa[:,:,2]=np.ones((3,3))*5
# # a=np.array(range(27))
# aa=np.float32(np.reshape(aa,(3,3,3,1)))
# print(aa[1,:,:,0])
# W=tf.convert_to_tensor(aa)
# input=tf.convert_to_tensor(im)
# out_backpropF=tf.convert_to_tensor(np.float32(np.ones((1,16,16,1))))
# conv = tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding='SAME')
# gradconvW = tf.nn.conv2d_backprop_filter(input=input, filter_sizes=[3,3,3,1], out_backprop=out_backpropF, strides=[1,1,1,1],
#                                          padding='SAME')

with tf.Session() as sess:
    uim=sess.run(upim)
    #ww=sess.run(gradconvW)

print(np.sum(uim,axis=(2,3)))



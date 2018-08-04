import tensorflow as tf
import h5py
import numpy as np
import pylab as py
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

im=np.ones((16,16,3))
im[:,:,1]=np.ones((16,16))*2
im[:,:,2]=np.ones((16,16))*3
im=np.float32(np.reshape(im,(1,16,16,3)))
print(im[0,:,:,1])
tf.reset_default_graph()

aa=np.ones((3,3,3))*3
aa[:,:,1]=np.ones((3,3))*4
aa[:,:,2]=np.ones((3,3))*5
# a=np.array(range(27))
aa=np.float32(np.reshape(aa,(3,3,3,1)))
print(aa[1,:,:,0])
W=tf.convert_to_tensor(aa)
input=tf.convert_to_tensor(im)
out_backpropF=tf.convert_to_tensor(np.float32(np.ones((1,16,16,1))))
conv = tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding='SAME')
gradconvW = tf.nn.conv2d_backprop_filter(input=input, filter_sizes=[3,3,3,1], out_backprop=out_backpropF, strides=[1,1,1,1],
                                         padding='SAME')

with tf.Session() as sess:
    cc=sess.run(conv)
    ww=sess.run(gradconvW)

print(cc)



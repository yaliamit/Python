import tensorflow as tf
import h5py
import numpy as np
import pylab as py
filename = '../_CIFAR100/cifar10_train.hdf5'
print(filename)
f = h5py.File(filename, 'r')
key = list(f.keys())[0]
# Get the data
tr = f[key]
print('tr',tr.shape)
key = list(f.keys())[1]
tr_lb=f[key]
train_data=np.float32(tr[0:45000])/255.
#py.imshow(train_data[0])
#py.show()
tf.reset_default_graph()
a=np.array(range(27))
aa=np.float32(np.reshape(a,(3,3,3,1)))
print(aa[1,:,:,0])
W=tf.convert_to_tensor(aa)
input=tf.convert_to_tensor(train_data[0:3])
out_backpropF=tf.convert_to_tensor(train_data[4:7][:,:,:,0:1])
conv = tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding='SAME')
gradconvW = tf.nn.conv2d_backprop_filter(input=input, filter_sizes=[3,3,3,1], out_backprop=out_backpropF, strides=[1,1,1,1],
                                         padding='SAME')

with tf.Session() as sess:
    cc=sess.run(conv)
    ww=sess.run(gradconvW)

print(cc)
print(train_data.shape)



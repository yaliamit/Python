import tensorflow as tf
import h5py
import numpy as np
import pylab as py
import keras
from keras.layers.convolutional import UpSampling2D
import Conv_net_gpu
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

# im=np.ones((2,4,4,3))
# im[0,:,:,1]=np.random.rand(4,4) #ones((4,4))*2
# im[0,:,:,2]=np.random.rand(4,4) #ones((4,4))*3
#im=np.float32(np.reshape(im,(1,4,4,3)))

# In[4]:

def local_pooling(input,pool_size, stride):



    shp=input.shape.as_list()
    paddings=np.int32(np.zeros((4,2)))
    paddings[1,:]=[pool_size,pool_size]
    paddings[2,:]=[pool_size,pool_size]
    pad=tf.convert_to_tensor(paddings)
    pinput=tf.pad(input,paddings=pad)
    ll=[]
    for j in range(pool_size):
        for k in range(pool_size):
            ll.append(tf.manip.roll(pinput,shift=[-j,-k],axis=[1,2]))

    shifted_images=tf.stack(ll)

    shifted_images = shifted_images[:,:, pool_size:pool_size + shp[1], pool_size:pool_size + shp[2], :]
    checker = np.zeros(shifted_images.shape.as_list(), dtype=np.bool)
    checker[:, :, 0::stride, 0::stride, :] = True
    Tchecker = tf.convert_to_tensor(checker)
    maxes = tf.reduce_max(shifted_images, axis=0)
    cmaxes=tf.tile(tf.expand_dims(maxes,0),[pool_size*pool_size,1,1,1,1])
    pooled = maxes[:,0::stride,0::stride,:]


    JJJ=tf.logical_and(tf.equal(cmaxes,shifted_images),Tchecker)
    jjj=[]
    for j in range(pool_size):
        for k in range(pool_size):
            jjj.append(tf.manip.roll(JJJ[j*pool_size+k,:,:,:,:],shift=[j,k],axis=[1,2]))
    UUU=tf.stack(jjj)
    mask=tf.reduce_sum(tf.cast(UUU,dtype=tf.int32),axis=0)

    return(pooled,mask)


im=np.random.rand(2,6,6,3)


tf.reset_default_graph()
input=tf.convert_to_tensor(im)

pool_size=2
stride=2
inputp=local_pooling(input,pool_size,stride)

with tf.Session() as sess:
    pooled,mask=sess.run(inputp)

print(im[0,:,:,0])

print(pooled[0,:,:,0])
print(mask[0,:,:,0])


print('done')
# upim=UpSampling2D(size=[2,2])(input)
# # aa=np.ones((3,3,3))*3
# # aa[:,:,1]=np.ones((3,3))*4
# # aa[:,:,2]=np.ones((3,3))*5
# # # a=np.array(range(27))
# # aa=np.float32(np.reshape(aa,(3,3,3,1)))
# # print(aa[1,:,:,0])
# # W=tf.convert_to_tensor(aa)
# # input=tf.convert_to_tensor(im)
# # out_backpropF=tf.convert_to_tensor(np.float32(np.ones((1,16,16,1))))
# # conv = tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding='SAME')
# # gradconvW = tf.nn.conv2d_backprop_filter(input=input, filter_sizes=[3,3,3,1], out_backprop=out_backpropF, strides=[1,1,1,1],
# #                                          padding='SAME')
#
# with tf.Session() as sess:
#     uim=sess.run(upim)
#     #ww=sess.run(gradconvW)
#
# print(np.sum(uim,axis=(2,3)))



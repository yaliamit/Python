import tensorflow as tf
import numpy as np
from keras import backend as K
from keras.layers.convolutional import UpSampling2D
import sys

def conv_layer(input,batch_size,nonlin_scale,filter_size=[3,3],num_features=[1],prob=[1.,-1.],scale=0, Win=None, Rin=None):

    # Get number of input features from input and add to shape of new layer
    shape=filter_size+[input.get_shape().as_list()[-1],num_features]
    shapeR=shape
    if (prob[1]==-1.):
        shapeR=[1,1]
    if (Rin is None):
        R = tf.get_variable('R',shape=shapeR)
    else:
        R = tf.get_variable('R',initializer=Rin)
    if (Win is None):
        W = tf.get_variable('W',shape=shape) # Default initialization is Glorot (the one explained in the slides)
    else:
        W = tf.get_variable('W',initializer=Win)
    input = tf.reshape(input, shape=[batch_size]+input.get_shape().as_list()[1:])

    #b = tf.get_variable('b',shape=[num_features],initializer=tf.zeros_initializer)
    conv = tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding='SAME')
    if (scale>0):
        conv = tf.clip_by_value(nonlin_scale * conv, -1., 1.)
    return(conv)

def grad_conv_layer(batch_size,below, back_propped, current, W, R, scale):
    w_shape=W.shape
    strides=[1,1,1,1]
    back_prop_shape=[-1]+(current.shape.as_list())[1:]
    out_backprop=tf.reshape(back_propped,back_prop_shape)
    if (scale>0):
        on_zero = K.zeros_like(out_backprop)
        out_backprop = scale * K.tf.where(tf.equal(tf.abs(current), 1.), on_zero, out_backprop)
    gradconvW=tf.nn.conv2d_backprop_filter(input=below,filter_sizes=w_shape,out_backprop=out_backprop,strides=strides,padding='SAME')
    input_shape=[batch_size]+(below.shape.as_list())[1:]

    filter=W
    if (len(R.shape.as_list())==4):
        filter=R
    print('input_sizes',input_shape,'filter',filter.shape.as_list(),'out_backprop',out_backprop.shape.as_list())
    gradconvx=tf.nn.conv2d_backprop_input(input_sizes=input_shape,filter=filter,out_backprop=out_backprop,strides=strides,padding='SAME')

    return gradconvW, gradconvx

def fully_connected_layer(input,batch_size,nonlin_scale, num_features,prob=[1.,-1.], scale=0,Win=None,Rin=None):
    # Make sure input is flattened.
    flat_dim=np.int32(np.array(input.get_shape().as_list())[1:].prod())
    input_flattened = tf.reshape(input, shape=[batch_size,flat_dim])
    shape=[flat_dim,num_features]
    shapeR=shape
    if (prob[1]==-1.):
        shapeR=[1]
    if (Rin is None):
        R_fc = tf.get_variable('R',shape=shapeR)
    else:
        R_fc = tf.get_variable('R',initializer=Rin)
    if (Win is None):
        W_fc = tf.get_variable('W',shape=shape)
    else:
        W_fc = tf.get_variable('W',initializer=Win)
    fc = tf.matmul(input_flattened, W_fc)
    if (scale>0):
        fc = tf.clip_by_value(nonlin_scale * fc, -1., 1.)
    return(fc)




def grad_fully_connected(below, back_propped, current, W, R, scale=0):

    belowf=tf.contrib.layers.flatten(below)
    # Gradient of weights of dense layer
    if (scale>0):
        on_zero = K.zeros_like(back_propped)
        back_propped = scale * K.tf.where(tf.equal(tf.abs(current), 1.), on_zero, back_propped)
    gradfcW=tf.matmul(tf.transpose(belowf),back_propped)
    # Propagated error to conv layer.
    filter=W
    if (len(R.shape.as_list())==2):
        filter=R
    gradfcx=tf.matmul(back_propped,tf.transpose(filter))

    return gradfcW, gradfcx

def sparse_fully_connected_layer(input,batch_size,nonlin_scale, num_units, num_features,prob=[1.,-1.], scale=0,Win=None,Rin=None, Fin=None):
    # Make sure input is flattened.
    input_shape=input.get_shape().as_list()
    flat_dim=np.int32(np.prod(input_shape[1:]))
    input_flattened = tf.reshape(input, shape=[batch_size,flat_dim])
    shape=[flat_dim,num_units]
    shapeR=shape
    if (prob[1]==-1.):
        shapeR=[1]

    if (Rin is None):
        R_dims = tf.get_variable('Rdims',initializer=[1])
        R_vals = tf.get_variable('Rvals',shape=[1,1])
        R_inds = tf.get_variable('Rinds',shape=[1,1])
    else:
        R_dims = tf.get_variable('Rdims',initializer=Rin.dense_shape)
        R_vals = tf.get_variable('Rvals',initializer=Rin.values)
        R_inds = tf.get_variable('Rinds',initializer=Rin.indices)
    if (Win is None):
       sys.exit('There should have been a precomputed W filter before this layer was created')
    else:
        W_dims = tf.get_variable('Wdims',initializer=Win.dense_shape)
        W_vals = tf.get_variable('Wvals',initializer=Win.values)
        W_inds = tf.get_variable('Winds',initializer=Win.indices)
    F_dims = tf.get_variable('Fdims',initializer=Fin.dense_shape)
    F_vals = tf.get_variable('Fvals',initializer=Fin.values)
    F_inds = tf.get_variable('Finds',initializer=Fin.indices)

    fc = tf.transpose(tf.sparse_tensor_dense_matmul(tf.SparseTensor(indices=W_inds,values=W_vals,dense_shape=W_dims),tf.transpose(input_flattened)))
    fc = tf.reshape(fc,input_shape[0:3]+[num_features,])
    # If non-linearity
    if (scale>0):
        fc = tf.clip_by_value(nonlin_scale * fc, -1., 1.)
    return(fc)

# Gradient for sparse fully connected.
def grad_sparse_fully_connected(below, back_propped, current, F_inds, F_vals, F_dims, W_inds, R_inds, scale=0):

    # Flatten whatever is coming from below
    belowf=tf.contrib.layers.flatten(below)
    # If non-linearity
    if (scale>0):
        on_zero = K.zeros_like(back_propped)
        back_propped = scale * K.tf.where(tf.equal(tf.abs(current), 1.), on_zero, back_propped)
    # Flatten whatever is coming from abbove
    back_proppedf=tf.contrib.layers.flatten(back_propped)
    # Get the active indices from below and above for the out-product gradient.
    below_list=tf.gather(belowf,W_inds[:,1],axis=1)
    back_propped_list=tf.gather(back_proppedf,W_inds[:,0],axis=1)
    gradfcW=tf.reduce_sum(tf.multiply(below_list,back_propped_list),axis=0)
    # Same for the gradient of R.
    if (R_inds is not None):
        below_list = tf.gather(belowf, R_inds[:, 1], axis=1)
        back_propped_list = tf.gather(back_proppedf, R_inds[:, 0], axis=1)
        gradfcR = tf.reduce_sum(tf.multiply(below_list, back_propped_list), axis=0)
    else:
        gradfcR=gradfcW
    # Finds,F_vals, F_dims stores the transpose of either W or R depending if we're doing non-symmetric
    filter=tf.SparseTensor(indices=F_inds,values=F_vals,dense_shape=F_dims)
    gradfcx=tf.transpose(tf.sparse_tensor_dense_matmul(filter,tf.transpose(back_proppedf)))
    gradfcx=tf.reshape(gradfcx,below.shape)
    return gradfcW, gradfcx, gradfcR

def MaxPoolingandMask(input,pool_size, stride):

# We are assuming 'SAME' padding with 0's.
    shp=input.shape.as_list()
    ll=[]
    # Create pool_size x pool_size shifts of the data stacked on top of each pixel
    # to represent the pool_size x pool_size window with that pixel as upper left hand corner
    for j in range(pool_size):
        for k in range(pool_size):
            pp=np.int64(np.zeros((4,2)))
            pp[1,:]=[0,j]
            pp[2,:]=[0,k]
            input_pad=tf.pad(input,pp)
            ll.append(input_pad[:,j:j+shp[1],k:k+shp[2],:])

    shifted_images = tf.stack(ll, axis=0)
    # Get the max in each stack
    maxes = tf.reduce_max(shifted_images, axis=0, name='Max')
    # Expand to the stack
    cmaxes = tf.tile(tf.expand_dims(maxes, 0), [pool_size * pool_size, 1, 1, 1, 1])
    # Get the pooled maxes by jumping strides
    pooled = tf.strided_slice(maxes, [0, 0, 0, 0], shp, strides=[1, stride, stride, 1], name='Max')
    # Create the checker board filter based on the stride
    checker = np.zeros([pool_size*pool_size,shp[0]]+shp[1:4], dtype=np.bool)
    checker[:, :, 0::stride, 0::stride, :] = True
    Tchecker = tf.convert_to_tensor(checker) #get_variable(initializer=checker,name='checker',trainable=False)

    # Filter the cmaxes and checker
    JJJ=tf.cast(tf.logical_and(tf.equal(cmaxes,shifted_images),Tchecker),dtype=tf.float32)
    # Reshift the cmaxes so that the stack for each pixel has true for indices corresonding to the upper left corner of each window
    # that used that pixel as the max.
    jjj=[]
    for j in range(pool_size):
        for k in range(pool_size):
            ind = j * pool_size + k
            pp = np.int64(np.zeros((4, 2)))
            pp[1, :] = [j,0]
            pp[2, :] = [k,0]
            jj_pad=tf.pad(JJJ[ind,:,:,:,:],paddings=pp)
            jjj.append(jj_pad[:,0:shp[1],0:shp[2],:])
    # This a pool_sizexpool_size stack of masks one for each location of ulc using the pixel as max.
    mask=tf.stack(jjj,axis=0, name='Equal')

    return pooled, mask


def grad_pool(back_propped, pool, mask, pool_size, stride):

        gradx_pool = tf.reshape(back_propped, [-1] + (pool.shape.as_list())[1:])
        ll = []
        gradx_pool=UpSampling2D(size=[stride,stride])(gradx_pool)
        shp = gradx_pool.shape.as_list()
        # Stack gradx values for different ulc of windows reaching pixel, add those flagged by mask.
        for j in range(pool_size):
            for k in range(pool_size):
                pp = np.int64(np.zeros((4, 2)))
                pp[1, :] = [j, 0]
                pp[2, :] = [k, 0]
                gradx_pool_pad=tf.pad(gradx_pool, paddings=pp)
                ll.append(gradx_pool_pad[:,0:shp[1],0:shp[2],:])

        shifted_gradx_pool = tf.stack(ll, axis=0)
        gradx=tf.reduce_sum(tf.multiply(shifted_gradx_pool,mask),axis=0)

        return gradx

def MaxPoolingandMask_disjoint_fast(inputs, pool_size, strides,
                          padding='SAME'):

        pooled = tf.nn.max_pool(inputs, ksize=pool_size, strides=strides, padding=padding)
        upsampled = UpSampling2D(size=strides[1:3])(pooled)
        input_shape=inputs.get_shape().as_list()
        upsampled_shape=upsampled.get_shape().as_list()
        if (input_shape != upsampled_shape):
            pads=np.zeros((4,2))
            for i in range(4):
                pads[i,1]=upsampled_shape[i]-input_shape[i]
            pinput=tf.pad(inputs,paddings=pads)
        else:
            pinput=inputs
        indexMask = K.tf.equal(pinput, upsampled)
        #assert indexMask.get_shape().as_list() == inputs.get_shape().as_list()
        return pooled,indexMask



def grad_pool_disjoint_fast(back_propped,pool,mask,pre,pool_size):

        gradx_pool=tf.reshape(back_propped,[-1]+(pool.shape.as_list())[1:])
        on_success = UpSampling2D(size=pool_size)(gradx_pool)
        on_fail = K.zeros_like(on_success)
        gradx=K.tf.where(mask, on_success, on_fail)
        dim = on_success.get_shape().as_list()
        predim = pre.get_shape().as_list()
        if (dim !=predim):
            gradx=gradx[:,0:predim[1],0:predim[2],:]
        return gradx


def real_drop(parent, drop,batch_size):
    U = tf.less(tf.random_uniform([batch_size] + (parent.shape.as_list())[1:]),drop)
    Z = tf.zeros_like(parent)
    fac = tf.constant(1.) / (1. - drop)
    drop = K.tf.where(U, Z, parent * fac)
    return drop
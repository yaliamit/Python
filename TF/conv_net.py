import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline


def conv_layer(input, filter_size=[3, 3], num_features=[1], prob=[1., -1.]):
    # Get number of input features from input and add to shape of new layer
    shape = filter_size + [input.get_shape().as_list()[-1], num_features]
    shapeR = shape
    if (prob[1] == -1.):
        shapeR = [1, 1]
    R = tf.get_variable('R', shape=shapeR)
    W = tf.get_variable('W', shape=shape)  # Default initialization is Glorot (the one explained in the slides)

    # b = tf.get_variable('b',shape=[num_features],initializer=tf.zeros_initializer)
    conv = tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding='SAME')
    conv = tf.clip_by_value(conv, -1., 1.)

    return (conv)


def grad_conv_layer(below, back_propped, current, W, R):
    w_shape = W.shape
    strides = [1, 1, 1, 1]
    back_prop_shape = [-1] + (current.shape.as_list())[1:]
    out_backprop = tf.reshape(back_propped, back_prop_shape)
    on_zero = K.zeros_like(out_backprop)
    out_backpropF = K.tf.where(tf.equal(tf.abs(current), 1.), on_zero, out_backprop)
    gradconvW = tf.nn.conv2d_backprop_filter(input=below, filter_sizes=w_shape, \
                                             out_backprop=out_backpropF, \
                                             strides=strides, \
                                             padding='SAME')
    input_shape = [batch_size] + (below.shape.as_list())[1:]

    filter = W
    if (len(R.shape.as_list()) == 4):
        filter = R
    gradconvx = tf.nn.conv2d_backprop_input(input_sizes=input_shape, filter=filter, out_backprop=out_backpropF,
                                            strides=strides, padding='SAME')

    return gradconvW, gradconvx
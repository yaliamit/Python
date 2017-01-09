import os
import numpy as np
import sys
import time
import theano
import theano.tensor as T
import lasagne
from lasagne.layers.dnn import Conv2DDNNLayer as Conv2DLayer
from lasagne.layers.dnn import MaxPool2DDNNLayer as MaxPool2DLayer

from dataPreparation import load_data

def iterate_minibatches(inputs, targets, batchsize, classNum = 10, shuffle=False):
    targets = np.array(targets, dtype = np.int32)
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        binary_targets = np.zeros((batchsize, classNum))
        binary_targets[np.arange(batchsize),targets[excerpt]] = 1
        yield inputs[excerpt], np.array(binary_targets,dtype = np.float32)

def main():
    # Load the dataset
    print("Loading data...")

    X_train, y_train, X_test, y_test = load_data("/X_train.npy", "/Y_train.npy", "/X_test.npy", "/Y_test.npy")

    input_var = T.ftensor4('inputs')
    target_var = T.fmatrix('targets')

    network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                        input_var=input_var, name = 'input_layer')

    network_reshape = lasagne.layers.ReshapeLayer(network, shape=([0], 784), name = 'reshape_layer')

    labelInput = lasagne.layers.InputLayer(shape=(None, 10),
                                        input_var=target_var)
    network = lasagne.layers.ConcatLayer(
            [network_reshape, labelInput], axis = 1)
    network = lasagne.layers.MultiGaussianMixture(network, num_components = 5, n_classes = 10 , name = 'output_layer')

    loss = lasagne.layers.get_output(network)
    loss.name = 'loss'
    loss_mean = loss.mean()
    loss_mean.name = 'loss_mean'

    params = lasagne.layers.get_all_params(network, trainable=True)
    print(params)
    #updates = lasagne.updates.nesterov_momentum(
    #        loss_mean, params, learning_rate=0.001, momentum=0.9)
    gparams = T.grad(loss_mean, params)


    updates = [
        (param, param - 0.1 * gparam)
        for param, gparam in zip(params, gparams)
     ]

    train_fn = theano.function([input_var, target_var], loss_mean, updates=updates)
    X_train_zero = np.array(X_train[y_train == 0], dtype = np.float32)
    y_train_zero = np.array(y_train[y_train == 0], dtype = np.float32)

    for epoch in range(2):
        train_err = 0
        batch_index = 0
        for batch in iterate_minibatches(X_train_zero, y_train_zero, 100, 10, shuffle=True):
            inputs, targets = batch
            current_result = train_fn(inputs, targets)
            train_err += current_result
            print("-------------")
            print(current_result)
            print(np.mean(lasagne.layers.get_all_param_values(network)[2][0]))
            print("-------------")
            batch_index += 1
        print(train_err)

    learnedWeights = lasagne.layers.get_all_param_values(network)
    import amitgroup.plot as gr
    gr.images(lasagne.layers.get_all_param_values(network)[0][0].reshape(5,28,28))
    #np.save("../data/gaussianDigits.npy", learnedWeights) 

if __name__ == "__main__":
    main()

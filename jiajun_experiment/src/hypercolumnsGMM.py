# This file is to extract hypercolumns from a arbitrary set of layers, and then build a GMM for each of the class
# Then we try to use the GMM to do likelihood 

import os
import numpy as np
import sys
import time
import theano
import theano.tensor as T
import lasagne
import scipy
#import amitgroup.plot as gr
import pnet
from lasagne.layers.dnn import Conv2DDNNLayer as Conv2DLayer
from lasagne.layers.dnn import MaxPool2DDNNLayer as MaxPool2DLayer
from dataPreparation import load_data
from CNNForMnist import build_cnn


def extract_hypercolumn_batch(model, layer_indexes, instance):
    batch_size = 5000
    n_batches = instance.shape[0] // batch_size
    data_feature = []
    for i in range(n_batches):
        print i
        data_feature.append(extract_hypercolumn(model, layer_indexes, instance[i * batch_size : (i + 1) * batch_size]))
    data_feature = np.vstack(data_feature)
    return data_feature

def extract_hypercolumn(model, layer_indexes, instance):
    all_layers = lasagne.layers.get_all_layers(model)
    layers = [lasagne.layers.get_output(all_layers[li]) for li in layer_indexes]
    get_feature = theano.function([all_layers[0].input_var], layers, allow_input_downcast=False)
    all_feature_maps = get_feature(instance)
    hypercolumn_result = [() for i in range(instance.shape[0])]


    # rows of hypercolumn 
    for feature_maps in all_feature_maps:
        # different data
        for data_index in range(instance.shape[0]):
            each_feature_maps = feature_maps[data_index]
            hypercolumns = []
            for convMap in each_feature_maps:
                upscaled = scipy.misc.imresize(convMap, size = (28, 28), mode = 'F', interp = 'bilinear')
                hypercolumns.append(upscaled)
            hypercolumn_result[data_index] += (np.asarray(hypercolumns),)

    for hyperColumn_index in range(instance.shape[0]):
        hypercolumn_result[hyperColumn_index] = np.vstack(hypercolumn_result[hyperColumn_index])
    return np.asarray(hypercolumn_result)
    

def main():
    X_train, y_train, X_test, y_test = load_data("/X_train.npy", "/Y_train.npy", "/X_test.npy", "/Y_test.npy")
    print("building function...")
    inputData = T.tensor4('inputs')
    cnnMnist = build_cnn(inputData)
    weightsOfParameters = np.load("../data/mnist_CNN_params_drop_out.npy")
    lasagne.layers.set_all_param_values(cnnMnist, weightsOfParameters)
    print("extracting all the hypercolumns...")
    
    train_hypercolumn = extract_hypercolumn_batch(cnnMnist, [3], X_train)    
    test_hypercolumn = extract_hypercolumn_batch(cnnMnist, [3], X_test) 
    print(train_hypercolumn.shape) 
    #gr.images(test_hypercolumn[0], vmin = None, vmax = None, cmap = None, colorbar = True, fileName = "./png/testhypercolumn.png") 
    print("training GMM for the hypercolumns...")
    objectModelLayer = pnet.HyperMixtureClassificationLayer(n_components = 5, min_prob = 0.0001, mixture_type = "gaussian")
    objectModelLayer.train(train_hypercolumn, y_train)
    print("finish training...")

    print("object model classification accuracy: ", np.mean(objectModelLayer.extract(test_hypercolumn) == y_test))

if __name__ == "__main__":
    main() 

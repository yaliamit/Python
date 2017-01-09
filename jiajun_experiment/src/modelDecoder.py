# This file is to use the model built on the bottleneck layer, and then produce the model on the finer grid.

# we first need to guild the gaussian/bernoulli mixture model on the bottleneck layer, and then multiply this with the matricies (that are used to mimic the convolution operations and pooling operations)

import os
import numpy as np
import sys
import time
import theano
import theano.tensor as T
import lasagne
from lasagne.layers.dnn import Conv2DDNNLayer as Conv2DLayer
from lasagne.layers.dnn import MaxPool2DDNNLayer as MaxPool2DLayer
from CNNForMnist import build_cnn, load_data
import pnet
import amitgroup.plot as gr
import matplotlib.pylab as plt
from buildLikelihoodDetector import encoder_extraction, extract
from convolutionAsMatrix import *

def multiplyDiagonal(leftMatrix, diagonalMatrix):
    result_temp = np.zeros((leftMatrix.shape[0], diagonalMatrix.shape[0]))
    for i in range(leftMatrix.shape[0]):
        result_temp[i] = leftMatrix[i] * np.sqrt(diagonalMatrix)
        
    return result_temp

def load_conv_weights(weightsFile):
    weightsOfParameters = np.load(weightsFile)[8:]
    return weightsOfParameters[0], weightsOfParameters[1]


def build_test_decoder(input_var = None):
    network = lasagne.layers.InputLayer(shape=(None, 512), input_var = input_var)
    network = lasagne.layers.ReshapeLayer(network, shape = (([0], 32, 4, 4)))
    network = lasagne.layers.Upscale2DLayer(network, 2)
    network = Conv2DLayer(network, num_filters = 32, filter_size = 5, pad = 'full', b = None, nonlinearity = None)
    network = lasagne.layers.Upscale2DLayer(network, 2)
    network = Conv2DLayer(network, num_filters = 1, filter_size = 5, pad = 'full', b = None, nonlinearity = None)
    network = lasagne.layers.ReshapeLayer(network, shape = (([0], -1)))
    return network

def test_equal(input_value, compareResult, weight_value = None):
    input_var = T.fmatrix('input variable')
    decoder = build_test_decoder(input_var)
    if len(lasagne.layers.get_all_params(decoder)) > 0:
        print(len(weight_value))
        lasagne.layers.set_all_param_values(decoder, weight_value)
    networkOutput = lasagne.layers.get_output(decoder, input_var)
    outputFunction = theano.function([input_var], networkOutput)
    expectedValue = outputFunction(input_value)
    print(expectedValue)
    print("===========")
    print(compareResult)
    print np.sum((expectedValue.reshape(expectedValue.shape[0], -1) - compareResult.reshape(compareResult.shape[0], -1)) ** 2)
    model_mean = expectedValue.reshape(10, 5, 28, 28)
    plt.figure(figsize = (10,20))
    plt.imshow(model_mean[0, 0], cmap = plt.cm.gray, interpolation='nearest')
    plt.savefig("./png/generatedGaussianModel_test.png")
    

def main():
    
    print("load data...")
    X_train, y_train, X_test, y_test = load_data("/X_train.npy", "/Y_train.npy", "/X_test.npy", "/Y_test.npy")
    print("bulding function...")

    extract_function = encoder_extraction(extraction_layer = 6, weights_file = "../data/mnist_autoencoder_params_encoder_linear_decoder_no_bias.npy")

    print("feature extraction...")
    X_train_feature = extract(X_train, extract_function)
    X_test_feature = extract(X_test, extract_function)
   
    print("train llh model...")
    objectModelLayer = pnet.MixtureClassificationLayer(n_components = 5, min_prob = 0.0001, mixture_type = "gaussian")
    #objectModelLayer.train(X_train_feature, y_train)
    #np.save("../data/object_model_rectify_activation_10_class_gaussian_no_bias.npy", objectModelLayer._modelinstance)
    objectModelLayer._modelinstance = np.load("../data/object_model_rectify_activation_10_class_gaussian_no_bias.npy")

    #objectModelLayer._modelinstance = np.load("")
    print("object model classification accuracy: ", np.mean(objectModelLayer.extract(X_train_feature) == y_train))
    # Shape of the model: (N_y, N_c, N_p)
    gaussianModel = objectModelLayer._modelinstance
    gaussianMeans = np.array([model.means_ for model in gaussianModel])
    gaussianCovars = np.array([model.covars_ for model in gaussianModel])

    conv1, conv2 = load_conv_weights("../data/mnist_autoencoder_params_encoder_linear_decoder_no_bias.npy") 

    upscaleMatrix1 = upscaleMatrix(featureShape = (32, 4, 4), stride = (2, 2))
    convMatrix1 = findMatrix(conv1, originalSize = 8, flip_filters = False)


    upscaleMatrix2 = upscaleMatrix(featureShape = (32, 12, 12), stride = (2, 2))
    convMatrix2 = findMatrix(conv2, originalSize= 24, flip_filters = False)
    
    #model_mean = np.dot(gaussianMeans, fc1)
    #model_sigma = np.dot(np.dot(np.transpose(fc1), gaussianCovars), fc1)
    #model_mean = np.dot(model_mean, fc2)
    #model_sigma = np.dot(np.dot(np.transpose(fc2), model_sigma), fc2)
    print("finish constructing the matrices")

    print(gaussianMeans.shape, gaussianCovars.shape)
    
    gaussianCovars = gaussianCovars.reshape(-1, gaussianCovars.shape[2])
    
    covars = np.zeros((gaussianCovars.shape[0], gaussianCovars.shape[1], 1))

    covars[:, :, 0] = np.sqrt(gaussianCovars)

    gaussianMeans = gaussianMeans.reshape(-1, gaussianMeans.shape[-1])
    
    model_mean = np.array([np.dot(upscaleMatrix1, gaussianMeans[i]) for i in range(gaussianMeans.shape[0])])
    model_sigma = np.array([np.dot(upscaleMatrix1, covar) for covar in covars])
    print(model_mean.shape, model_sigma.shape)

    #test_equal(np.array(gaussianMeans, dtype = np.float32), np.array(model_mean, dtype = np.float32), weight_value = [conv1])


    model_mean = np.array([np.dot(convMatrix1, model_mean[i]) for i in range(model_mean.shape[0])])
    model_sigma = np.array([np.dot(convMatrix1, covar) for covar in model_sigma])
    print(model_mean.shape, model_sigma.shape)
    
    #test_equal(np.array(gaussianMeans, dtype = np.float32), np.array(model_mean, dtype = np.float32), [conv1])
    
    model_mean = np.array([np.dot(upscaleMatrix2, model_mean[i]) for i in range(model_mean.shape[0])])
    model_sigma = np.array([np.dot(upscaleMatrix2, covar) for covar in model_sigma])
    print(model_mean.shape, model_sigma.shape)
    
    #test_equal(np.array(gaussianMeans, dtype = np.float32), np.array(model_mean, dtype = np.float32), [conv1])



    model_mean = np.array([np.dot(convMatrix2, model_mean[i]) for i in range(model_mean.shape[0])])
    model_sigma = np.array([np.dot(convMatrix2, covar) for covar in model_sigma])
    print(model_mean.shape, model_sigma.shape)
    print(model_mean.shape)
    

    test_equal(np.array(gaussianMeans, dtype = np.float32), np.array(model_mean, dtype = np.float32), weight_value = [conv1, conv2])
    
    result_object = []
    
    model_mean = model_mean.reshape(10, 5, 28, 28)
    for i in range(10):
        object_images = model_mean[i]
        object_images = np.hstack(object_images)
        result_object.append(object_images)
    
    result_object = np.vstack(result_object)
    plt.figure(figsize = (10,20))
    plt.imshow(result_object, cmap = plt.cm.gray, interpolation='nearest')
    plt.savefig("./png/generatedGaussianModel.png")

if __name__ == "__main__":
    main()

    
    
    
    









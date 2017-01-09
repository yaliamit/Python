# This file is used to build a gaussian/bernoulli mixture model based on the features encoded by the first four layers (conv2d + pool2d + conv2d + pool2d).

# After a gaussian/bernoulli mixture model is build, we try to use it on a large image as a object detector. We use non-maximal surpression to locate the candidate locations
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
from convolutionAutoEncoder import build_autoencoder
import pnet
import amitgroup.plot as gr

INT_MIN = -10000000

def createSampleTest(nSample = 1, sampleSize = 40):
    X_train, y_train, X_test, y_test = load_data("/X_train.npy", "/Y_train.npy", "/X_test.npy", "/Y_test.npy")
    data = np.zeros((nSample, sampleSize, sampleSize), dtype = np.float32)
    for i in range(nSample):
        xLoc = np.random.randint(sampleSize - 28)
        yLoc = np.random.randint(sampleSize - 28)
        data[i, xLoc : xLoc + 28, yLoc : yLoc + 28] = X_train[i]
    return data
    


def encoder_extraction(extraction_layer = 6, weights_file = "../data/mnist_CNN_params_drop_out.npy"):
    inputData = T.tensor4('inputs')
    autoencoderNN = build_cnn(inputData)
    print(lasagne.layers.get_all_layers(autoencoderNN))
    #autoencoderNN = build_autoencoder(inputData)
    #weightsOfParameters = np.load("../data/mnist_CNN_params_sigmoid.npy")
    #weightsOfParameters = np.load("../data/mnist_CNN_params.npy")
    weightsOfParameters = np.load(weights_file)
    lasagne.layers.set_all_param_values(autoencoderNN, weightsOfParameters)
    networkOutputLayer = lasagne.layers.get_all_layers(autoencoderNN)[extraction_layer]
    extractedFeature = lasagne.layers.get_output(networkOutputLayer, deterministic = True)
    return theano.function([inputData], extractedFeature)
    
def extract(data, extract_function):
    batch_size = 200
    n_batches = data.shape[0] // batch_size
    data_feature = []
    for i in range(n_batches):
        data_feature.append(extract_function(np.array(data[i * batch_size : (i + 1) * batch_size], dtype = np.float32)))
    data_feature = np.vstack(data_feature)
    return data_feature

def reprocess_data_for_extraction(inputX, inputShape = 28):
    returnResult = np.zeros((inputX.shape[0], inputX.shape[1] - inputShape + 1, inputX.shape[2] - inputShape + 1, inputShape, inputShape))
    for i in range(inputX.shape[1] - inputShape + 1):
        for j in range(inputX.shape[2] - inputShape + 1):
            returnResult[:,i, j] = inputX[:, slice(i, i + inputShape), slice(j, j + inputShape)]
    return returnResult


def nonMaximalSupress(llh, windowSize = 5):
    nSample, X, Y = llh.shape
    for x in range(X - windowSize + 1):
        for y in range(Y - windowSize + 1):
            window = llh[:, x:x+windowSize, y:y+windowSize]
            localMax = np.amax(window, axis = (1, 2))
            maxCoord = np.unravel_index(np.argmax(window.reshape(-1, windowSize * windowSize), axis = 1) + [i * windowSize * windowSize for i in range(nSample)], (nSample, windowSize, windowSize))
            regionCoord = tuple(maxCoord[1:] + np.array((x, y)).reshape(2,1))
            llh[:, x:x+windowSize, y:y+windowSize] = INT_MIN
            llh[tuple((maxCoord[0], regionCoord[0], regionCoord[1]))] = localMax
    return llh
            

def main():
    current_model = "train"

    print("load data...")
    X_train, y_train, X_test, y_test = load_data("/X_train.npy", "/Y_train.npy", "/X_test.npy", "/Y_test.npy")
    print("bulding function...")
    #extract_function = encoder_extraction(extraction_layer = 6, weights_file = "../data/mnist_autoencoder_params_encoder_linear_decoder_no_bias.npy")
    extract_function = encoder_extraction()
    print("feature extraction...")
    X_train_feature = extract(X_train, extract_function)
    X_test_feature = extract(X_test, extract_function)
    print(X_train_feature.shape)
    print(X_test_feature.shape)
    
    print("train llh model...")
    objectModelLayer = pnet.MixtureClassificationLayer(n_components = 5, min_prob = 0.0001, mixture_type = "gaussian")
    if current_model == "train":
        objectModelLayer.train(X_train_feature, y_train)
        np.save("../data/object_model_rectify_activation_10_class_gaussian_fc.npy", objectModelLayer._models)

        print("object model classification accuracy: ", np.mean(objectModelLayer.extract(X_test_feature) == y_test))
        
        data = createSampleTest(nSample = 1)
        gr.images(data[0])
        processedData = reprocess_data_for_extraction(data)
        processedData_reshape = processedData.reshape(-1, 1, 28, 28)
        processedData_reshape_feature = extract_function(np.array(processedData_reshape,dtype = np.float32))
        llh_for_data = objectModelLayer.score(processedData_reshape_feature).reshape(processedData.shape[:3] + (10, ))
        print llh_for_data.shape
        print(np.max(llh_for_data, axis = -1))
        print(np.argmax(llh_for_data, axis = -1))
        print(np.argmax(llh_for_data, axis = -1)[nonMaximalSupress(np.max(llh_for_data, axis = -1), windowSize = 5) != INT_MIN])
        print (y_train[:1]) 
    else:
        model_means = np.load("../data/reconstructModel_mean.npy")
        model_sigmas = np.load("../data/reconstructModel_sigma.npy")
        model_weights = np.load("../data/reconstructModel_weights.npy")
        objectModelLayer._modelinstance = []
        for i in range(10):
            from sklearn.mixture import GMM
            mm = GMM(n_components = 5, n_iter = 20, n_init = 1, random_state = 0, covariance_type = 'full')
            mm.covars_ = model_sigmas[i]
            mm.means_ = model_means[i].reshape(5, -1)
            mm.weights_ = model_weights[i]
            objectModelLayer._modelinstance.append(mm)
        objectModelLayer._models = model_means
        print("calculating classification accuracy...")
        print("object model classification accuracy: ", np.mean(objectModelLayer.extract(X_train.reshape(X_train.shape[0], -1)) == y_train))
        
        data = createSampleTest(nSample = 1)
        gr.images(data[0])
        processedData = reprocess_data_for_extraction(data)
        processedData_reshape = processedData.reshape(-1, 784)
        llh_for_data = objectModelLayer.score(processedData_reshape)#.reshape(processedData.shape[:3] + (10, ))
        print llh_for_data.shape
        #print(np.max(llh_for_data, axis = -1))
        #print(np.argmax(llh_for_data, axis = -1))
        #print(np.argmax(llh_for_data, axis = -1)[nonMaximalSupress(np.max(llh_for_data, axis = -1), windowSize = 5) != INT_MIN])
        #print (y_train[:1]) 


if __name__ == "__main__":
    main() 

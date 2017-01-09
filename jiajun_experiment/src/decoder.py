import os
import numpy as np
import sys
import time
import theano
import theano.tensor as T
import lasagne
import matplotlib.pylab as plt

from lasagne.layers.dnn import Conv2DDNNLayer as Conv2DLayer
from lasagne.layers.dnn import MaxPool2DDNNLayer as MaxPool2DLayer
from CNNForMnist import load_data
from convolutionAutoEncoder import build_autoencoder


## Decode after two layers

def build_decoder(input_var = None):
    
    network = lasagne.layers.InputLayer(shape=(None, 32, 4, 4),
                                        input_var=input_var)
    
    network = lasagne.layers.DenseLayer(
            #lasagne.layers.dropout(network, p=.5),
            network,
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify,
            #nonlinearity = lasagne.nonlinearities.sigmoid,
            )

    network = lasagne.layers.DenseLayer(
            network,
            num_units=512,
            nonlinearity = None,
            #nonlinearity=lasagne.nonlinearities.rectify,
            #nonlinearity = lasagne.nonlinearities.sigmoid
            )

    network = lasagne.layers.ReshapeLayer(
            network,
            shape = (([0], 32, 4, 4)))

    network = lasagne.layers.Upscale2DLayer(
            network,
            2 )
    network = Conv2DLayer(
            network, num_filters = 32, filter_size = 5, pad = 'full',
            #nonlinearity = lasagne.nonlinearities.sigmoid,
            #nonlinearity = lasagne.nonlinearities.rectify,              
            nonlinearity = None)

    network = lasagne.layers.Upscale2DLayer(
            network,
            2 )

    network = Conv2DLayer(
            network, num_filters = 1, filter_size = 5, pad = 'full',
            #nonlinearity = lasagne.nonlinearities.sigmoid,
            #nonlinearity = lasagne.nonlinearities.rectify,
            nonlinearity = None)

    network = lasagne.layers.ReshapeLayer(
            network, shape = (([0], -1)))

    return network



def main():
    #autoencoderWeights = np.load("../data/mnist_clutter_autoencoder_params_sigmoid.npy")
    autoencoderWeights = np.load("../data/mnist_autoencoder_params_encoder_linear_decoder.npy")
    inputVar = T.tensor4('input variable')
    decoderNetwork = build_decoder(inputVar)
    lasagne.layers.set_all_param_values(decoderNetwork, autoencoderWeights[4:])

    reconstructed_data = lasagne.layers.get_output(decoderNetwork, inputVar)
    reconstruct = theano.function([inputVar], reconstructed_data)

    #objectModels = np.load("../data/object_model_sigmoid.npy")
    objectModels = np.load("../data/object_model_rectify_activation_10_class_gaussian.npy")
    result_object = []
    for i in range(10):
        object_images = reconstruct(np.array(objectModels[i], dtype = np.float32)).reshape(5, 28, 28)
        object_images = np.hstack(object_images)
        result_object.append(object_images)
    result_object = np.vstack(result_object)
    plt.figure(figsize = (10,20))
    plt.imshow(result_object, cmap = plt.cm.gray, interpolation='nearest')
    plt.show()


if __name__ == "__main__":
    main()

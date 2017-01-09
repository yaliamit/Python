import os
import numpy as np
import sys
import time
import theano
import theano.tensor as T
import lasagne
from lasagne.layers.dnn import Conv2DDNNLayer as Conv2DLayer
from lasagne.layers.dnn import MaxPool2DDNNLayer as MaxPool2DLayer
from CNNForMnist import load_data

def build_autoencoder(input_var=None):
    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    # Expert note: Lasagne provides alternative convolutional layers that
    # override Theano's choice of which implementation to use; for details
    # please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.

    # Max-pooling layer of factor 2 in both dimensions:
    network = MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W = lasagne.init.GlorotUniform()
            )

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            #lasagne.layers.dropout(network, p=.5),
            network,
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify,
            )

    network = lasagne.layers.DenseLayer(
            #lasagne.layers.dropout(network, p=.5),
            network,
            num_units=512,
            nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.ReshapeLayer(
            network,
            shape = (([0], 32, 4, 4)))

    network = lasagne.layers.Upscale2DLayer(
            network,
            2 )
    network = Conv2DLayer(
            network, num_filters = 32, filter_size = 5, pad = 'full',
            nonlinearity = lasagne.nonlinearities.rectify,
            )
    
    network = lasagne.layers.Upscale2DLayer(
            network,
            2 )

    network = Conv2DLayer(
            network, num_filters = 1, filter_size = 5, pad = 'full',
            nonlinearity = lasagne.nonlinearities.rectify,
            )

    network = lasagne.layers.ReshapeLayer(
            network, shape = (([0], -1)))

    return network

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]



def main(num_epochs = 50):
    # We fix the encoder in this experiment
    print ("loading data...")
    X_train, y_train, X_test, y_test = load_data("/X_train.npy", "/Y_train.npy", "/X_test.npy", "/Y_test.npy")
    #X_train, y_train, X_test, y_test = load_data("/cluttered_train_x.npy", "/cluttered_train_y.npy", "/cluttered_test_x.npy", "/cluttered_test_y.npy", dataset = "MNIST_CLUTTER")

    height = 28
    width = 28
    
    input_var = T.tensor4('inputs')
    target_var_original = T.tensor4('targets')
    target_var = target_var_original.reshape((-1, height * width)) 
    
    print ("build network...")    
    network = build_autoencoder(input_var)
    reconstructed_train = lasagne.layers.get_output(network, input_var)

    L = T.mean(lasagne.objectives.squared_error(target_var, reconstructed_train), axis = 1)
    #L = -T.mean(target_var * T.log(reconstructed_train) + (1 - target_var) * T.log(1 - reconstructed_train), axis=1)
    cost = T.mean(L)
    
    decoder_params = lasagne.layers.get_all_params(network, trainable = True)[6:]
    print(decoder_params)
    updates = lasagne.updates.nesterov_momentum(
        cost, decoder_params, learning_rate = 0.03, momentum = 0.95)
    #gparams = T.grad(cost, decoder_params)
    #updates = [(param, param - 0.01 * gparam) for param, gparam in zip(decoder_params, gparams)]

    reconstructed_test = lasagne.layers.get_output(network, deterministic=True)
    #reconstructed_test_loss = T.mean(-T.sum(target_var * T.log(reconstructed_test) + (1 - target_var) * T.log(1 - reconstructed_test), axis = 1))
    reconstructed_test_loss = T.mean(lasagne.objectives.squared_error(target_var, reconstructed_test), axis = 1)
    reconstructed_cost = T.mean(reconstructed_test_loss)

    print ("brewing functions....")
    train_fn = theano.function([input_var, target_var_original], cost, updates = updates)
    val_fn = theano.function([input_var, target_var_original], reconstructed_cost)

    print ("intialize encoder parameters")
    # We fix the encoder in this experiment
    #encoder_weights = np.load("../data/mnist_clutter_CNN_params_sigmoid.npy")
    #encoder_weights = np.load("../data/mnist_CNN_params_sigmoid.npy")
    #encoder_weights = np.load("../data/mnist_CNN_params.npy")
    encoder_weights = np.load("../data/mnist_CNN_params_drop_out.npy")
    #Try to get the first four layers of [conv, pool, conv, pool]
    encoder_model = lasagne.layers.get_all_layers(network)[5]
    print(lasagne.layers.get_all_layers(network))
    # The learned weights also contains the parameters for the softmax layer. Need to crop that out.
    #Set that equals to 2 since there is no bias; If there is bias, it should be 4
    lasagne.layers.set_all_param_values(encoder_model, encoder_weights[:6])
    

    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(30):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        # Here the training and targets are the same: X_train, since we are doing autoencoding.
        for batch in iterate_minibatches(X_train, X_train, 100, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1


        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))

        if epoch % 1 == 0:
            # After training, we compute and print the test error:
            test_err = 0
            test_batches = 0
            for batch in iterate_minibatches(X_test, X_test, 500, shuffle=False):
                inputs, targets = batch
                err = val_fn(inputs, targets)
                test_err += err
                test_batches += 1
            print("Final results:")
            print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    weightsOfParams = lasagne.layers.get_all_param_values(network)
    #np.save("../data/mnist_clutter_autoencoder_params_sigmoid.npy", weightsOfParams) 
    #np.save("../data/mnist_autoencoder_params_sigmoid.npy", weightsOfParams) 
    #np.save("../data/mnist_autoencoder_params_encoder_linear_decoder_no_bias.npy", weightsOfParams) 
    np.save("../data/mnist_autoencoder_params_NY.npy", weightsOfParams) 

if __name__ == '__main__':
    main() 

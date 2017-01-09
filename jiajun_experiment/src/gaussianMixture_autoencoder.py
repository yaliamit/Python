# This file is a editted version of https://github.com/Lasagne/Lasagne/blob/master/examples/mnist.py
# Use Lasagne for digit recognition using  MNIST dataset.
import os
import numpy as np
import sys
import time
import theano
import theano.tensor as T
import lasagne
#from lasagne.layers.dnn import Conv2DDNNLayer as Conv2DLayer
#from lasagne.layers.dnn import MaxPool2DDNNLayer as MaxPool2DLayer
from lasagne.layers import Conv2DLayer as Conv2DLayer
from lasagne.layers import MaxPool2DLayer as MaxPool2DLayer
import amitgroup.plot as gr
os.chdir("../src/")

def build_cnn(input_var=None, input_label=None):

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                        input_var=input_var)
    
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            #nonlinearity=lasagne.nonlinearities.sigmoid,
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
            #nonlinearity=lasagne.nonlinearities.sigmoid
            )
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network_output = lasagne.layers.DenseLayer(
            network,
            #lasagne.layers.dropout(network, p=.5),
            num_units=256,
            #nonlinearity=lasagne.nonlinearities.sigmoid
            nonlinearity=lasagne.nonlinearities.rectify,
            )
    
    encoder_output = lasagne.layers.reshape(network, shape = ([0], 512))
    
#     network = lasagne.layers.ConcatLayer(
#             [network_output, labelInput], axis = 1)
    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    #network = lasagne.layers.DenseLayer(
    #        lasagne.layers.dropout(network, p=.5),
    #        num_units=10,
    #        nonlinearity=lasagne.nonlinearities.softmax)
    
    gaussian_output = lasagne.layers.MultiGaussianMixture(encoder_output, num_components = 5, n_classes = 10, sigma=lasagne.init.Constant(1))

    network = lasagne.layers.ReshapeLayer(encoder_output, shape = (([0], 32, 4, 4)))

    network = lasagne.layers.Upscale2DLayer(network, 2)

    network = Conv2DLayer(network, num_filters = 32, filter_size = 5, pad = 'full', nonlinearity = lasagne.nonlinearities.rectify)

    network = lasagne.layers.Upscale2DLayer(network, 2)

    decoder_output = Conv2DLayer(network, num_filters = 1, filter_size = 5, pad = 'full', nonlinearity = lasagne.nonlinearities.rectify)
    
    decoder_output = lasagne.layers.ReshapeLayer(decoder_output, shape = (([0], 28 * 28)))

    return gaussian_output, encoder_output, decoder_output



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
    from dataPreparation import load_data
    # Load the dataset
    print("Loading data...")
    X_train, y_train, X_test, y_test = load_data("/X_train.npy", "/Y_train.npy", "/X_test.npy", "/Y_test.npy")
    X_train = np.array(X_train, dtype = np.float32)
    y_train = np.array(y_train, dtype = np.float32)
    X_test = np.array(X_test, dtype = np.float32)
    y_test = np.array(y_test, dtype = np.float32)
    #X_train, y_train, X_test, y_test = load_data("/cluttered_train_x.npy", "/cluttered_train_y.npy", "/cluttered_test_x.npy", "/cluttered_test_y.npy", dataset = "MNIST_CLUTTER")

    dimension = np.prod(X_train.shape[1:])
    # Prepare Theano variables for inputs and targets
    input_var = T.ftensor4('inputs')
    target_var = T.fmatrix('targets')
    target_input_var = input_var.reshape((-1, dimension))
    # Create neural network model (depending on first command line parameter)

    gaussian_output, encoder_output, decoder_output = build_cnn(input_var, target_var)
    
    #weightsOfParams = np.load("../data/mnist_autoencoder_params_encoder_linear_decoder.npy")
    #lasagne.layers.set_all_param_values(fc, weightsOfParams[:4]) 

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    llh_output = lasagne.layers.get_output(gaussian_output)
    loss_1 = lasagne.objectives.multi_negative_llh(llh_output, target_var)
    fc_output = lasagne.layers.get_output(encoder_output)
    loss_mean_1 = T.mean(loss_1)
    reconstruction = lasagne.layers.get_output(decoder_output)
    loss_mean_2 = T.mean(T.mean(lasagne.objectives.squared_error(reconstruction, target_input_var), axis = 1))

    alpha = T.scalar('alpha', dtype=theano.config.floatX)
    combination = T.exp(alpha)/(1 + T.exp(alpha))
    loss_mean_burn = combination * loss_mean_1 / 10000 + (1 - combination) * loss_mean_2
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    #params = list(set(lasagne.layers.get_all_params(decoder_output, trainable=True) + lasagne.layers.get_all_params(gaussian_output, trainable=True)))
    params = lasagne.layers.get_all_params(decoder_output, trainable=True)
    gaussianMixtureParameters = lasagne.layers.get_all_params(gaussian_output, trainable = True)
    gaussianParam = []
    for param in gaussianMixtureParameters:
        if param not in params:
            gaussianParam.append(param)
    print(params)
    print("model built")
    #updates = lasagne.updates.nesterov_momentum(
    #        loss_mean_2, params, learning_rate=0.1, momentum=0.9)
    gparams = T.grad(loss_mean_burn, params + gaussianParam)
    print(gparams)
    updates = []

    for param, gparam in zip(params + gaussianParam, gparams):
        if param in params:
            updates.append((param, param - 0.0001 * gparam))
        elif param in gaussianParam:
            updates.append((param,  param - 0.01 * gparam))
    print(updates)
    #0.000001
    # updates = [(param, param - 0.0000001 * gparam)
    #     for param, gparam in zip(params[:4], gparams[:4])] + [(params[4], params[4] - 0.01 * gparams[4])]


    #updates = [(param, param - 0.01 * gparam)
    #    for param, gparam in zip(params[:8], gparams[:8])] + [(param, param - 0.01 * gparam) for param, gparam in zip(params[8:], gparams[8:])]
    

    #updates = [(param, param - 0.05 * gparam) for param, gparam in zip(params, gparams)]
    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_llh_output = lasagne.layers.get_output(gaussian_output, deterministic=True)
    test_loss = lasagne.objectives.multi_negative_llh(test_llh_output, target_var)
    test_loss_mean = T.mean(test_loss)
    test_reconstruction = lasagne.layers.get_output(decoder_output, deterministic = True)
    test_loss_reconstruction_mean = T.mean(T.mean(lasagne.objectives.squared_error(test_reconstruction, target_input_var), axis = 1))
    
    test_total_loss = combination * test_loss_mean / 10000 + (1 - combination) * test_loss_reconstruction_mean
    # As a bonus, also create an expression for the classification accuracy:

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    # + [update_param for update_param in gparams]
    train_fn = theano.function([input_var, target_var, alpha], [loss_mean_burn, llh_output, loss_mean_1, loss_mean_2], updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var, alpha], [test_total_loss, test_llh_output, test_loss_mean, test_loss_reconstruction_mean, test_reconstruction])

    # Finally, launch the training loop.


    print("Starting training...")
    # We iterate over epochs:

    loss_combination = 0
    num_epochs = 2000
    for epoch in range(num_epochs):
        #loss_combination = loss_combination + 0.02
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_reconstruction_err = 0
        train_llh_err = 0
        train_batches = 0
        start_time = time.time()
        batchIndex = 0
        for batch in iterate_minibatches(X_train, y_train, 50, 10, shuffle=True):
            inputs, targets = batch
            #current_loss_mean, current_loss, fc_output = train_fn(inputs, targets)
            current_result = train_fn(inputs, targets, loss_combination)

    #         if batchIndex % 2000 == 0:
    #             print(current_result)
            batchIndex = batchIndex + 1
            train_err += current_result[0]
            train_reconstruction_err += current_result[3]
            train_llh_err += current_result[2]
            train_batches += 1
            break


        if epoch % 100 == 0: 
            
            print(loss_combination)
                # Then we print the results for this epoch:
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, num_epochs, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
            print("  train llh loss:\t\t{:.6f}".format(train_llh_err / train_batches))
            print("  training reconstruction loss:\t\t{:.6f}".format(train_reconstruction_err / train_batches))
            #print(lasagne.layers.get_all_param_values(network)[-3])
            print("--")
        
            # After training, we compute and print the test error:
            test_err = 0
            test_llh_err = 0
            test_reconstruction_err = 0
            test_batches = 0
            accurate = 0
            for batch in iterate_minibatches(X_test, y_test, 50, 10, shuffle=False):
                inputs, targets = batch
                valuationResult = val_fn(inputs, targets, loss_combination)
                err = valuationResult[0]
                test_err += err
                test_llh_err += valuationResult[2]
                test_reconstruction_err += valuationResult[3]
                test_batches += 1
                accurate += np.sum(np.argmax(targets, axis = 1) == np.argmax(valuationResult[1], axis = 1))
            if epoch % 500 == 0:
                gr.images(valuationResult[4].reshape(-1, 28, 28), show = False, fileName = "/Users/jiajunshen/Desktop/%d.png"%epoch)
            print("Final results:")
            print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
            print("  test llh loss:\t\t\t{:.6f}".format(test_llh_err / test_batches))
            print("  test reconstruction loss:\t\t\t{:.6f}".format(test_reconstruction_err / test_batches))
            print("Test Accuracy: ", accurate / 10000.0)


            # Optionally, you could now dump the network weights to a file like this:
            # np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
            #
            # And load them again later on like this:
            # with np.load('model.npz') as f:
            #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            # lasagne.layers.set_all_param_values(network, param_values)
    # weightsOfParams = lasagne.layers.get_all_param_values(network)
    # #np.save("../data/mnist_clutter_CNN_params_sigmoid.npy", weightsOfParams)
    # #np.save("../data/mnist_CNN_params_sigmoid.npy", weightsOfParams)
    # np.save("../data/mnist_CNN_gaussian.npy", weightsOfParams)


if __name__ == "__main__":
    main()

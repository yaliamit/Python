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


def build_cnn(input_var=None, input_label=None):

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                        input_var=input_var)
    labelInput = lasagne.layers.InputLayer(shape=(None, 10),
                                        input_var=input_label)
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
    network = lasagne.layers.DenseLayer(
            network,
            #lasagne.layers.dropout(network, p=.5),
            num_units=256,
            #nonlinearity=lasagne.nonlinearities.sigmoid
            nonlinearity=lasagne.nonlinearities.rectify,
            )
    network = lasagne.layers.ConcatLayer(
            [network, labelInput], axis = 1)
    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    #network = lasagne.layers.DenseLayer(
    #        lasagne.layers.dropout(network, p=.5),
    #        num_units=10,
    #        nonlinearity=lasagne.nonlinearities.softmax)
    network = lasagne.layers.MultiGaussianMixture(network, num_components = 5, n_classes = 10)
    return network



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


def main(model='mlp', num_epochs=500):
    
    os.chdir("../src/")

    from dataPreparation import load_data
    # Load the dataset
    print("Loading data...")
    
    X_train, y_train, X_test, y_test = load_data("/X_train.npy", "/Y_train.npy", "/X_test.npy", "/Y_test.npy")
    X_train = np.array(X_train, dtype = np.float32)
    y_train = np.array(y_train, dtype = np.float32)
    X_test = np.array(X_test, dtype = np.float32)
    y_test = np.array(y_test, dtype = np.float32)
    #X_train, y_train, X_test, y_test = load_data("/cluttered_train_x.npy", "/cluttered_train_y.npy", "/cluttered_test_x.npy", "/cluttered_test_y.npy", dataset = "MNIST_CLUTTER")

    # Prepare Theano variables for inputs and targets
    input_var = T.ftensor4('inputs')
    target_var = T.fmatrix('targets')

    # Create neural network model (depending on first command line parameter)

    network = build_cnn(input_var, target_var)

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    loss = lasagne.layers.get_output(network)
    loss = loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    print(params)
    print("model built")
    #updates = lasagne.updates.nesterov_momentum(
    #        loss, params, learning_rate=0.0001, momentum=0.9)
    gparams = T.grad(loss, params)


    updates = [
        (param, param - 1 * gparam)
        for param, gparam in zip(params, gparams)
     ] 

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_loss = lasagne.layers.get_output(network, deterministic=True)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], [loss,] + [update for update in gparams], updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], test_loss)

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(50):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        batchIndex = 0
        for batch in iterate_minibatches(X_train, y_train, 50, 10, shuffle=True):
            inputs, targets = batch
            current_result = train_fn(inputs, targets)
            if batchIndex % 1 == 0:
                print(current_result[0])
            batchIndex = batchIndex + 1
            train_err += current_result[0]            
            train_batches += 1
            

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        #print(lasagne.layers.get_all_param_values(network)[-3])
        print("--")
        if epoch % 5 == 6: 
            # After training, we compute and print the test error:
            test_err = 0
            test_batches = 0
            for batch in iterate_minibatches(X_test, y_test, 50, 10, shuffle=False):
                inputs, targets = batch
                err = val_fn(inputs, targets)
                test_err += err
                test_batches += 1
            print("Final results:")
            print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
        

            # Optionally, you could now dump the network weights to a file like this:
            # np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
            #
            # And load them again later on like this:
            # with np.load('model.npz') as f:
            #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            # lasagne.layers.set_all_param_values(network, param_values)
    weightsOfParams = lasagne.layers.get_all_param_values(network)
    #np.save("../data/mnist_clutter_CNN_params_sigmoid.npy", weightsOfParams)
    #np.save("../data/mnist_CNN_params_sigmoid.npy", weightsOfParams)
    np.save("../data/mnist_CNN_gaussian.npy", weightsOfParams)



if __name__ == '__main__':
    main()

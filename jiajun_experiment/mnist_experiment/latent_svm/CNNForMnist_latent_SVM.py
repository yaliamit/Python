# This file is a editted version of https://github.com/Lasagne/Lasagne/blob/master/examples/mnist.py
# Use Lasagne for digit recognition using  MNIST dataset.
import os
import numpy as np
import sys
import time
import theano
import theano.tensor as T
import lasagne
import cv2
from lasagne.layers.dnn import Conv2DDNNLayer as Conv2DLayer
from lasagne.layers.dnn import MaxPool2DDNNLayer as MaxPool2DLayer
# from lasagne.layers import Conv2DLayer
# from lasagne.layers import MaxPool2DLayer
from dataPreparation import load_data

# Turn nw*h to n*w*h*nRotation
def rotateImage_batch(image, num_rotation=8):
    result_list = []
    for i in range(num_rotation):
        angle = (360.0 / num_rotation) * i
        rotated_inputs = np.array([rotateImage(image[j], angle) for j in range(image.shape[0])], dtype = np.float32)            
        result_list.append(rotated_inputs) 

    return np.array(result_list, dtype = np.float32)

def rotateImage(image, angle):
  if len(image.shape) == 3:
        image = image[0]
  image_center = tuple(np.array(image.shape)/2)
  rot_mat = cv2.getRotationMatrix2D(image_center,angle,1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape,flags=cv2.INTER_LINEAR)
  return np.array(result[:, :], dtype = np.float32)

def build_cnn(input_var=None):

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 1, 40, 40),
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
            W = lasagne.init.GlorotUniform()
            #nonlinearity=lasagne.nonlinearities.sigmoid
            )
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            #network,
            num_units=256,
            #nonlinearity=lasagne.nonlinearities.sigmoid
            nonlinearity=lasagne.nonlinearities.rectify,
            )

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            #network,
            num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)

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


def extend_image(inputs, size = 40):
    extended_images = np.zeros((inputs.shape[0], 1, size, size), dtype = np.float32)
    margin_size = (40 - inputs.shape[2]) / 2
    extended_images[:, :, margin_size:margin_size + inputs.shape[2], margin_size:margin_size + inputs
.shape[3]] = inputs
    return extended_images


def main(model='mlp', num_epochs=500):
    # Load the dataset
    print("Loading data...")
    num_per_class = 100
    print("Using %d per class" % num_per_class) 
    
    X_train, y_train, X_test, y_test = load_data("/X_train.npy", "/Y_train.npy", "/X_test.npy", "/Y_test.npy")
    X_train_final = []
    y_train_final = []
    for i in range(10):
        X_train_class = X_train[y_train == i]
        # permutated_index = np.random.permutation(X_train_class.shape[0])
        permutated_index = np.arange(X_train_class.shape[0])
        X_train_final.append(X_train_class[permutated_index[:100]])
        y_train_final += [i] * num_per_class
    X_train = np.vstack(X_train_final)
    y_train = np.array(y_train_final, dtype = np.int32) 
    
    X_train = extend_image(X_train, 40)
    X_test = extend_image(X_test, 40)
    #X_train, y_train, X_test, y_test = load_data("/cluttered_train_x.npy", "/cluttered_train_y.npy", "/cluttered_test_x.npy", "/cluttered_test_y.npy", dataset = "MNIST_CLUTTER")

    # Prepare Theano variables for inputs and targets
    nRotation = 8
    
    # The dimension would be (nRotation * n, w, h)
    input_var = T.tensor4('inputs')

    # The dimension would be (n, )
    vanilla_target_var = T.ivector('vanilla_targets')
    # The dimension would be (nRotation * n , )
    target_var = T.ivector('targets')

    # Create neural network model (depending on first command line parameter)
    
    network = build_cnn(input_var)
    
    saved_weights = np.load("../data/mnist_Chi_dec_100.npy")
    lasagne.layers.set_all_param_values(network, saved_weights)

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    # The dimension would be (nRotation * n, 10)
    predictions = lasagne.layers.get_output(network)

    # The dimension would be (nRotation * n, 10)
    one_hot_targets = T.extra_ops.to_one_hot(target_var, 10)

    # rests = [T.reshape(predictions[i][(1 - one_hot_targets).nonzero()], (-1, 9)) for i in range(nRotation)]
    
    rests = T.reshape(predictions[(1 - one_hot_targets).nonzero()], (nRotation, -1, 9))
    # a list of $nRotation tensor, each tensor is of shape (n, 1)
    rests = [T.max(rests[i], axis = 1) for i in range(nRotation)]

    latent_vector = T.argmin(T.as_tensor_variable(rests), axis = 0)

    # selected_index is a (n, nRotation) matrix of binary values.
    selected_index = T.extra_ops.to_one_hot(latent_vector, nRotation)

    predictions = T.reshape(predictions, (nRotation, -1, 10))

    predictions = T.as_tensor_variable(predictions).swapaxes(0, 1)

    predictions = predictions[selected_index.nonzero()]

    # loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = lasagne.objectives.multiclass_hinge_loss(predictions, vanilla_target_var)
    loss = loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    # updates = lasagne.updates.nesterov_momentum(
    #         loss, params, learning_rate=0.01, momentum=0.9)
    updates = lasagne.updates.adagrad(loss, params, learning_rate = 0.01)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_prediction = T.reshape(test_prediction,(nRotation, -1, 10))
    test_prediction_max = test_prediction.max(axis = 2)
    rotation_index = T.extra_ops.to_one_hot(T.argmax(test_prediction_max, axis = 0), nRotation)
    test_prediction = test_prediction.swapaxes(0, 1)[rotation_index.nonzero()]
    
    test_loss = lasagne.objectives.multiclass_hinge_loss(test_prediction, vanilla_target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), vanilla_target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var, vanilla_target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, vanilla_target_var], [test_loss, test_acc])

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, 100, shuffle=True):
            inputs, targets = batch
            inputs = inputs.reshape(100, 40, 40)
            inputs = rotateImage_batch(inputs, nRotation).reshape(100 * nRotation, 1, 40, 40)
            duplicated_targets = np.array([targets for i in range(nRotation)]).reshape(100 * nRotation,)
            train_err += train_fn(inputs, duplicated_targets, targets)
            train_batches += 1


        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))

        if epoch % 5 == 0: 
           # After training, we compute and print the test error:
            test_err = 0
            test_acc = 0
            test_batches = 0
            for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
                inputs, targets = batch
                inputs = inputs.reshape(500, 40, 40)
                inputs = rotateImage_batch(inputs, nRotation).reshape(500 * nRotation, 1, 40, 40)
                err, acc = val_fn(inputs, targets)
                test_err += err
                test_acc += acc
                test_batches += 1
            print("Final results:")
            print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
            print("  test accuracy:\t\t{:.2f} %".format(
                test_acc / test_batches * 100))

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
    #np.save("../data/mnist_CNN_params.npy", weightsOfParams)
    #np.save("../data/mnist_CNN_params_drop_out_semi_Chi_Dec7.npy", weightsOfParams)
    np.save("../data/mnist_CNN_params_drop_out_Chi_2017.npy", weightsOfParams)
    #np.save("../data/mnist_CNN_params_For_No_Bias_experiment_out.npy", weightsOfParams)



if __name__ == '__main__':
    main()

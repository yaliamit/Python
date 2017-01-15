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
from lasagne.regularization import regularize_layer_params_weighted, l2
# from lasagne.layers import Conv2DLayer
# from lasagne.layers import MaxPool2DLayer
from dataPreparation import load_data


# Turn nw*h to n*w*h*nRotation
def rotateImage_batch(image, num_rotation=16):
    result_list = []
    for i in range(num_rotation):
        # angle = (90.0 / num_rotation) * i - 45
        angle = (360.0 / num_rotation) * i
        rotated_inputs = np.array([rotateImage(image[j], angle) for j in range(image.shape[0])], dtype = np.float32)            
        result_list.append(rotated_inputs) 

    return np.array(result_list, dtype = np.float32)

# If we found out the rotation index of the image is i out of nRotation
# then we can unrotate the image, by the degree D:
# D = -(360 / num_rotation) * i
def unrotateImage(images, rotation_index, num_rotation = 16):
    result_list = []
    unrotated_results = np.array([rotateImage(images[i], \
        -(360.0 / num_rotation) * rotation_index[i]) for i in range(images.shape[0])], dtype = np.float32)

    return unrotated_results



def rotateImage(image, angle):
    if len(image.shape) == 3:
        image = image[0]
    if angle == 0:
        return image

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
    fc1  = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            #network,
            num_units=256,
            #nonlinearity=lasagne.nonlinearities.sigmoid
            nonlinearity=lasagne.nonlinearities.rectify,
            )

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    fc2  = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(fc1, p=.5),
            #network,
            num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)
    
    weight_decay_layers = {fc1:0.0, fc2:0.002}
    l2_penalty = regularize_layer_params_weighted(weight_decay_layers, l2)

    return fc2, l2_penalty




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


def main(model='mlp'):
    # Load the dataset
    print("Loading data...")
    # num_per_class = 100
    # print("Using %d per class" % num_per_class) 
    print("Using all the training data") 
    
    #X_train, y_train, X_test, y_test = load_data("/X_train.npy", "/Y_train.npy", "/X_test_rotated.npy", "/Y_test_rotated.npy")
    X_train, y_train, X_test, y_test = load_data("/mnistROT.npy", "/mnistROTLabel.npy", "/mnistROTTEST.npy", "/mnistROTLABELTEST.npy", "ROT_MNIST")
   
    X_train = extend_image(X_train, 40)
    X_test = extend_image(X_test, 40)

    # Prepare Theano variables for inputs and targets
    nRotation = 16
    
    # The dimension would be (nRotation * n, w, h)
    input_var = T.tensor4('inputs')

    # The dimension would be (n, )
    vanilla_target_var = T.ivector('vanilla_targets')
    # The dimension would be (nRotation * n , )
    target_var = T.ivector('targets')

    # Create neural network model (depending on first command line parameter)
    
    network, weight_decay = build_cnn(input_var)
    
    saved_weights = np.load("../data/mnist_CNN_params_drop_out_Chi_2017.npy")
    lasagne.layers.set_all_param_values(network, saved_weights)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_prediction = T.reshape(test_prediction,(nRotation, -1, 10))

    test_rotation = T.argmax(T.argmax(test_prediction, axis = 2), axis = 0)
    test_prediction = test_prediction.max(axis = 0)
     
    test_loss = lasagne.objectives.multiclass_hinge_loss(test_prediction, vanilla_target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), vanilla_target_var),
                      dtype=theano.config.floatX)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, vanilla_target_var], [test_loss, test_acc, test_rotation])

    # Finally, launch the training loop.
    print("Starting evaluating...")

    test_err = 0
    test_acc = 0
    test_batches = 0
    unrotated_images = []
    for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
        inputs, targets = batch
        inputs = inputs.reshape(500, 40, 40)
        inputs = rotateImage_batch(inputs, nRotation).reshape(500 * nRotation, 1, 40, 40)
        err, acc, test_rotation = val_fn(inputs, targets)
        unrotated_inputs = unrotateImage(inputs, test_rotation)
        unrotated_images.append(unrotated_inputs)
        test_err += err
        test_acc += acc
        test_batches += 1
    unrotated_images = np.vstack(unrotated_images)
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))
    np.save("./unrotated_images.npy", unrotated_images)
   


if __name__ == '__main__':
    main()

import numpy as np
import sys
import time
import theano
import theano.tensor as T
import lasagne
from lasagne.layers.dnn import Conv2DDNNLayer as Conv2DLayer
from lasagne.layers.dnn import MaxPool2DDNNLayer as MaxPool2DLayer
from CNNForMnist import build_cnn, load_data


# Build model using the same model we trained on MNIST_CLUTTER
reshaped_Data = T.tensor4('inputs')
classificatioNetwork = build_cnn(reshaped_Data)
weightsOfParameters = np.load("./mnist_clutter_CNN_params.npy")
lasagne.layers.set_all_param_values(classificatioNetwork, weightsOfParameters)

heatMap = lasagne.layers.get_output(classificatioNetwork).reshape((-1, 13, 13, 11))

inputX = np.zeros((1, 40, 40), dtype = np.float32)
def output_heapmap(inputX, inputShape = 28):
    returnResult = np.zeros((inputX.shape[0], inputX.shape[1] - inputShape + 1, inputX.shape[2] - inputShape + 1, inputShape, inputShape))
    for i in range(inputX.shape[1] - inputShape):
        for j in range(inputX.shape[2] - inputShape):
            returnResult[:,i, j] = inputX[:, slice(i, i + inputShape), slice(j, j + inputShape)]
    return returnResult

X_train, y_train, X_test, y_test = load_data("/X_train.npy", "/Y_train.npy", "/X_test.npy", "/Y_test.npy")

train_heatMap = theano.function([reshaped_Data], heatMap)

inputX[:,10:38, 10:38] = X_train[:1, 0]
x_crop = output_heapmap(inputX)
x_crop = x_crop.reshape((-1, 1, 28, 28))

result = train_heatMap(np.array(x_crop, dtype = np.float32))

labelResult = np.argmax(result[0], axis = 2)
print labelResult

print np.array(np.max(result[0], axis = 2))


#import amitgroup.plot as gr
#gr.images(inputX[0])
#
#for i in range(169):
#    gr.images(x_crop[i, 0])



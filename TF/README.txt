Code for running URFB, FRFB and regular SGD.

The code is run on the current machine as follows:

python run_conv.py _pars/fncrc OUT

The last argument is an output file, the second to last is a parameter file (it expects a .txt extension)
but don't add it in the command line.

If the machine has a gpu version of tensorflow it will use it.


The program implements back propagation and its modifications explicitly. The only use
of tensor flow gradient is for the last layer loss. All the rest of the back propagation is computed explicitly
using routines in the file Conv_layers so as to enable the URFB and FRFB algorithms. The only weight update
method implemented is straight SGD. For the convolution backpropagation we do use a tensorflow function called
conv2d_backprop_filter, conv2d_backprop_input.

The graph is created in the function `recreate_network` in the file Conv_net_gpu and the ops for
weight updates are created in the function `back_propagation` in the file Conv_net_gpu.

Some instructions on the parameter files:

This is an example parameter file. Most parameters are self explanatory.
The name comes first a semi-colin then the value. To enter a tuple of values use
parenthesese.

seed:45239
num_epochs:1
num_epochs_sparse:3 # Number of epochs after converting convolutional layers to sparse fully connected matrices.
data_set:mnist
batch_size:500
step_size:.1
sparse_step_size:.1
debug:False
num_train:5000
off_class_fac:1. # Factor multiplying 1/(C-1) weight on sum of non-class hinge losses.
hinge:1. # Use hinge loss with margin 1., False - use softmax.
force_global_prob:(1.,0.) # First coordinate sampling proportion of connectivity forward and backward.
                          # Second coordinate 1. - URFB, 0. FRFB, -1. SGD.
sparse:conv1R,  # Which convolutional layers to convert to sparse matrices (list must end with comma)
                # If a sparse field exists the network will move to convert convolutional layers after num_epochs.
                # If not it ends the training.
#non_trainable:conv1, # Which layers to stop training in sparse phase.
#re_randomize:conv1R,newdensp,newdensf # Which layers to reinitialize in sparse phase.
#shift:15 # Random geometric perturbations to apply to images.


# Below is the network architecture. - conv: if name contains conv it is a convolutional layer and expects num_filters,
#                                  filter_size, non-linearity is optional, if it is there it is always assumed tanh
#                                  although the tanh is redundant and the only non-linearity implemented is the saturated ramp from paper.
#                                  pool: if name contains pool this is a max pooling layer with pool_size and stride.
#                                  drop: if name contains drop it expects a drop parameter.
#                                  dens: if name contains dens its a dens layer and expects num_units.
#                                  concatsum: sum two earlier layers (for resnet type architectures). Parent names given in square brackets.
# All layers expect a parent. Final layer is indicated with the final field.
name:input1
name:conv1R;num_filters:8;filter_size:(5,5);non_linearity:tanh;parent:input1
name:conv1aR;num_filters:32;filter_size:(3,3);non_linearity:tanh;parent:conv1R
name:concatsum1;parent:[conv1,conv1aR]
name:pool1;pool_size:(3, 3);stride:(2, 2);parent:concatsum1
name:drop1;drop:.8;parent:pool1
name:densp;num_units:500;non_linearity:tanh;parent:drop1
name:drop2;drop:0.3;parent:densp
name:densf;num_units:10;non_linearity:soft_max;parent:drop2;final:final
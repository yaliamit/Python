"""A binary to train CIFAR-10 using a single GPU.

Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

import cifar10_merge
import cifar10_merge_input

import theano
import theano.tensor as T
import lasagne

batch_size = 400

train_dir = '/home/jiajun/MultipleDetection/cifar10_yali_2/cifar10_theano_train_merge'

max_steps = 100000

load_model = '/home/jiajun/MultipleDetection/cifar10_yali_2/cifar10_theano_train_adam/model_step60000.npy'

def train():
    """Train CIFAR-10 for a number of steps."""
    if os.path.isfile(FLAGS.load_model):
        all_weights = np.load(FLAGS.load_model) 
    else:
        print("Model file does not exist. Exiting....")
        return

    print("Build up the network")
    image_input_var = T.tensor4('original_inputs')
    target_var = T.ivector('targets')

    cnn_model, cnn_mid_output, weight_decay_penalty = cifar10_merge.build_cnn(image_input_var)

    original_model_mid_output = lasagne.layers.get_output(cnn_mid_output, image_input_var, deterministic = True)

    rotated_image_input_var = T.tensor4('rotated_image_input')
    
    rotated_cnn_model, rotated_model_mid, rotated_weight_penalty = \
        cifar10_merge.build_cnn(rotated_image_input_var)

    cnn_model_val, _, _ = cifar10_merge.build_cnn(rotated_image_input_var)    

    original_model_output_val = lasagne.layers.get_output(cnn_model_val, rotated_image_input_var, deterministic = True)


    rotated_model_mid_output = lasagne.layers.get_output(rotated_model_mid, rotated_image_input_var, deterministic = True)

    rotated_model_output = lasagne.layers.get_output(rotated_cnn_model, rotated_image_input_var, deterministic = True)


    lasagne.layers.set_all_param_values(cnn_model, all_weights)

    lasagne.layers.set_all_param_values(cnn_model_val, all_weights)

    rotated_net_weights_below_mid = lasagne.layers.get_all_param_values(rotated_model_mid)

    rotated_net_training_param = lasagne.layers.get_all_params(rotated_model_mid)

    lasagne.layers.set_all_param_values(rotated_cnn_model, all_weights)

    lasagne.layers.set_all_param_values(rotated_model_mid,
                                         rotated_net_weights_below_mid)

    L = T.mean(lasagne.objectives.squared_error(original_model_mid_output, rotated_model_mid_output), axis = 1)
    cost = T.mean(L)

    updates = lasagne.updates.adagrad(cost, rotated_net_training_param, learning_rate=0.1)

    # cross_entropy_loss = lasagne.objectives.categorical_crossentropy(model_output, target_var)

    # cross_entropy_loss_mean = cross_entropy_loss.mean()

    # loss = cross_entropy_loss_mean + weight_decay_penalty


    train_acc = T.mean(T.eq(T.argmax(rotated_model_output, axis = 1), target_var),
                       dtype=theano.config.floatX)

    original_model_acc = T.mean(T.eq(T.argmax(original_model_output_val, axis = 1), target_var),
                                dtype=theano.config.floatX)

    train_fn = theano.function(inputs = [image_input_var, rotated_image_input_var, target_var],
                               outputs = [cost, train_acc], updates = updates)

    val_fn = theano.function(inputs = [rotated_image_input_var, target_var],
                             outputs = [train_acc, original_model_acc])

    if os.path.isfile(os.path.join(FLAGS.train_dir, 'latest_model.txt')):
        weight_file = ""
        with open(os.path.join(FLAGS.train_dir, 'latest_model.txt'), 'r') as checkpoint_file:
            weight_file = checkpoint_file.read().replace('\n', '')
        print("Loading from: ", weight_file)
        model_weights = np.load(weight_file)
        lasagne.layers.set_all_param_values(rotated_cnn_model, model_weights)

    # Get images and labels for CIFAR-10.

    cifar10_data = cifar10_merge_input.load_cifar10()


    for step in xrange(FLAGS.max_steps):
        start_time = time.time()
        original_train_image, rotated_train_image, train_label = cifar10_data.train.next_batch(FLAGS.batch_size)
        end_time_1 = time.time() - start_time
        loss_value, train_acc = train_fn(original_train_image, rotated_train_image, train_label)
        duration = time.time() - start_time

        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

        if step % 10 == 0:
            num_examples_per_step = FLAGS.batch_size
            examples_per_sec = num_examples_per_step / duration
            sec_per_batch = float(duration)
            
            format_str = ('%s: step %d, loss = %.2f, train_acc = %.3f, (%.1f examples/sec; %.3f '
                          'sec/batch); %.3f sec/batch prepration')
            print (format_str % (datetime.now(), step, loss_value, train_acc,
                                 examples_per_sec, sec_per_batch, float(end_time_1)))
        
        if step % 20000 == 0 or (step + 1) == FLAGS.max_steps:
            checkpoint_path = os.path.join(FLAGS.train_dir, 'model_step%d.npy' % step)
            weightsOfParams = lasagne.layers.get_all_param_values(rotated_cnn_model)
            np.save(checkpoint_path, weightsOfParams)
            latest_model_path = os.path.join(FLAGS.train_dir, 'latest_model.txt')
            try:
                os.remove(latest_model_path)
            except OSError:
                pass
            latest_model_file = open(latest_model_path, "w")
            latest_model_file.write(checkpoint_path)
            latest_model_file.close()

        if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
            # Start to Evaluate
            original_test_image, rotated_test_image, test_label = cifar10_data.test.next_eval_batch(FLAGS.batch_size)
            total_acc_count = 0
            original_acc_count = 0
            total_count = 0

            print("Start Evaluating")

            while(rotated_test_image is not None):
                train_acc, original_acc = val_fn(rotated_test_image, test_label)
                total_acc_count += train_acc * rotated_test_image.shape[0]
                original_acc_count += original_acc * rotated_test_image.shape[0]
                total_count += rotated_test_image.shape[0]
                original_test_image, rotated_test_image, test_label = cifar10_data.test.next_eval_batch(FLAGS.batch_size)

            print("Final Accuracy: %.4f" % (float(total_acc_count / total_count)))
            print("Original Accuracy: %.4f" % (float(original_acc_count / total_count)))

def main(argv=None):  # pylint: disable=unused-argument
    train()


if __name__ == '__main__':
    main()

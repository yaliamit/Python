# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Routine for decoding the CIFAR-10 binary file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 32

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 10000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

import collections

def random_crop(images, dims):
    image_num, image_channel, image_height, image_width = images.shape
    assert image_height > dims[1] and image_width > dims[2] and image_channel == dims[0]
    limit_height = image_height - dims[1] + 1
    limit_width = image_width - dims[2] + 1
    offset_height = np.random.randint(low = 0, high = limit_height, size = image_num) % limit_height
    offset_width = np.random.randint(low = 0, high = limit_width, size = image_num) % limit_width
    cropped_image = [images[i, :, offset_height[i]:offset_height[i]+dims[1],
                            offset_width[i]:offset_width[i]+dims[2]]
                     for i in range(image_num)]
    return np.array(cropped_image)


def random_flip_left_right(images):
    image_num, image_channel, image_height, image_width = images.shape
    uniform_random = np.random.uniform(size = image_num)
    mirror = uniform_random > 0.5
    mirrored_image = [images[i, :, :, ::(mirror[i] *2 - 1)]
                     for i in range(image_num)]
    return np.array(mirrored_image)

def random_brightness(images, max_delta):
    image_num, image_channel, image_height, image_width = images.shape
    delta = np.random.uniform(-max_delta, max_delta, size = image_num)
    adjusted_image = np.array([images[i] + delta[i] for i in range(image_num)])
    adjusted_image = np.clip(adjusted_image, a_min = 0, a_max = 255)
    return adjusted_image

def random_contrast(images, lower, upper):
    image_num, image_channel, image_height, image_width = images.shape
    contrast_factor = np.random.uniform(lower, upper, size = image_num)
    adjusted_image = [[(images[i, j, :, :] - np.mean(images[i, j, :, :])) * \
                       contrast_factor[i] + np.mean(images[i, j, :, :]) for j in range(3)] \
                      for i in range(image_num)]
    adjusted_image = np.clip(adjusted_image, a_min = 0, a_max = 255)
    return np.array(adjusted_image)

def per_image_standardization(images):
    image_num, image_channel, image_height, image_width = images.shape
    standardized_image = np.array([(images[i] - np.mean(images[i])) / \
                                   max(np.std(images[i]), 1.0 / np.sqrt(image_channel * \
                                                                        image_height * \
                                                                        image_width)) \
                                   for i in range(image_num)])
    return standardized_image


def resize_with_crop_or_pad(images, image_size):
    image_num, image_channel, image_height, image_width = images.shape
    offset = np.abs(image_size - image_height) // 2
    if image_size <= image_height:
        final_image = np.array([images[i, :, offset:offset + image_size, offset: offset + image_size] for i in range(image_num)])
    else:
        images = np.rollaxis(images, 1, 4)
        print(images.shape)
        final_image = [cv2.copyMakeBorder(images[i], offset, offset, offset, offset, cv2.BORDER_REFLECT) for i in range(image_num)]
        final_image = np.rollaxis(np.array(final_image), 3, 1)
    return final_image 

class DataSet(object):
    def __init__(self,
                 images,
                 labels,
                 test=False,
                 distortion=True,
                 dtype=np.float32):
        self._num_examples = images.shape[0]
        self._images = np.array(images, dtype=dtype)
        self._images = self._images / 255.0
        # self._images = np.array(per_image_standardization(self._images), dtype=dtype)
        self._labels = np.array(labels, dtype=np.int32)
        self._index_in_epoch = 0
        self._index_in_eval_epoch = 0
        self._distortion = distortion

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    def data_prepration(self, images, image_size, distortion=True):
        if distortion:
            cropped_image = random_crop(images, [3, image_size, image_size])
            mirrored_image = random_flip_left_right(cropped_image)
            brightness_adjusted_image = random_brightness(mirrored_image, 64)
            contrast_adjuested_image = random_contrast(brightness_adjusted_image, 0.2, 1.8)
            before_standardized_image = contrast_adjuested_image
        else:
            before_standardized_image = resize_with_crop_or_pad(images, image_size)
        standardized_image = np.array(per_image_standardization(before_standardized_image), dtype=np.float32)
        return standardized_image

    def next_batch(self, batch_size, image_size=IMAGE_SIZE):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN:
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            start = 0
            self._index_in_epoch = batch_size
        end = self._index_in_epoch
        
        #return self.data_prepration(self._images[start:end], image_size, self._distortion),\
        #       self._labels[start:end]
        return self._images[start:end],\
               self._labels[start:end],\
               start

    def next_eval_batch(self, batch_size, image_size=IMAGE_SIZE, distort=False):
        start = self._index_in_eval_epoch
        self._index_in_eval_epoch += batch_size
        if start >= NUM_EXAMPLES_PER_EPOCH_FOR_EVAL:
            self._index_in_eval_epoch = 0
            return None, None
        else:
            end = self._index_in_eval_epoch
            #resize_image = resize_with_crop_or_pad(self._images[start:end], image_size)
            #standardized_image = per_image_standardization(resize_image)
            #return np.array(standardized_image, dtype=np.float32), self._labels[start:end]
            return self._images[start:end],\
                   self._labels[start:end]
            
    
            
    
def read_data_sets(data_dir, distortion=True, dtype=np.float32):
    train_images = np.array(np.load(os.path.join(data_dir, "cifar10TrainingData.npy")).reshape(50000, 3, 32, 32), dtype=dtype)
    train_labels = np.load(os.path.join(data_dir, "cifar10TrainingDataLabel.npy"))

    train_images_list = []
    train_labels_list = []

    for i in range(10):
        train_images_list.append(train_images[train_labels == i][:1000])
        train_labels_list += [i] * 1000
    index = np.arange(10000)
    train_images = np.vstack(train_images_list)[index]
    train_labels = np.array(train_labels_list, dtype = np.int32)[index]

    print(train_images.shape)
    print(train_labels.shape)

    test_images = np.array(np.load(os.path.join(data_dir, "cifar10TestingData.npy")).reshape(10000, 3, 32, 32), dtype=dtype)
    test_labels = np.load(os.path.join(data_dir, "cifar10TestingDataLabel.npy"))

    train = DataSet(train_images, train_labels, distortion=distortion)
    test = DataSet(test_images, test_labels, test=True)

    Datasets = collections.namedtuple('Datasets', ['train', 'test'])

    return Datasets(train = train, test = test)

def load_cifar10():
    return read_data_sets(os.environ['CIFAR10_DIR'])



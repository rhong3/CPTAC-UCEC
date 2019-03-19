"""
Data input preparation from decoding TFrecords, onehot encoding, augmentation, and batching 2.0

Created on 03/19/2019

@author: RH
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class DataSet(object):
    # bs is batch size; ep is epoch; images are images; mode is test/train; filename is tfrecords
    def __init__(self, bs, count, ep=1, images=None, mode=None, filename=None):
        self._batchsize = bs
        self._index_in_epoch = 0
        self._num_examples = count
        self._images = images
        self._mode = mode
        self._filename = filename
        self._epochs = ep

    # decoding tfrecords; return images and labels
    def decode(self, serialized_example):
        features = tf.parse_example(
            serialized_example,
            # Defaults are not specified since both keys are required.
            features={self._mode + '/image': tf.FixedLenFeature([], tf.string),
                      self._mode + '/label': tf.FixedLenFeature([], tf.int64), })

        image = tf.decode_raw(features[self._mode + '/image'], tf.float32)
        image = tf.reshape(image, [-1, 299, 299, 3])

        # Convert label from a scalar uint8 tensor to an int32 scalar.
        label = tf.cast(features[self._mode + '/label'], tf.int32)
        return image, label

    # augmentation including onehot encoding
    def augment(self, images, labels):

        angles = tf.round(tf.random_uniform([], 0, 4))
        images = tf.image.rot90(images, k=angles)
        images = tf.image.random_flip_left_right(images)
        images = tf.image.random_flip_up_down(images)
        images = tf.image.random_hue(images, 0.2)
        images = tf.image.random_brightness(images, 0.1)
        images = tf.image.random_contrast(images, 0.75, 1.5)
        images = tf.image.random_saturation(images, 0.75, 1.5)

        labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=4)

        return images, labels

    # onehot encoding only; for test set
    def onehot_only(self, images, labels):
        with tf.name_scope('onehot_only'):
            labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=4)
            labels = tf.cast(labels, tf.float32)
        return images, labels

    # dataset preparation; batching; Real test or not; train or test
    def data(self, Not_Realtest=True, train=True):
        batch_size = self._batchsize
        ep = self._epochs
        if Not_Realtest:
            filenames = tf.placeholder(tf.string, shape=None)
            dataset = tf.data.TFRecordDataset(filenames)
            dataset = dataset.repeat(ep)
            if train:
                batched_dataset = dataset.batch(batch_size, drop_remainder=True)
                batched_dataset = batched_dataset.map(self.decode)
                batched_dataset = batched_dataset.map(self.augment)
            else:
                batched_dataset = dataset.batch(batch_size, drop_remainder=False)
                batched_dataset = batched_dataset.map(self.decode)
                batched_dataset = batched_dataset.map(self.onehot_only)
            iterator = batched_dataset.make_initializable_iterator()
            return iterator, self._filename, filenames
        else:
            features_placeholder = tf.placeholder(self._images.dtype, self._images.shape)
            dataset = tf.data.Dataset.from_tensor_slices(features_placeholder)
            dataset = dataset.repeat(ep)
            batched_dataset = dataset.batch(batch_size, drop_remainder=False)
            iterator = batched_dataset.make_initializable_iterator()
            return iterator, self._images, features_placeholder

    @property
    def images(self):
        return self._images

    @property
    def num_examples(self):
        return self._num_examples

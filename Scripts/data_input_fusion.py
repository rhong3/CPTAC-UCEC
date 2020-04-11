"""
Data input preparation from decoding TFrecords, onehot encoding, augmentation, and batching 3.0 for fusion

Created on 11/06/2019

@author: RH
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class DataSet(object):
    # bs is batch size; ep is epoch; images are images; mode is test/train; filename is tfrecords
    def __init__(self, bs, count, ep=1, cls=2, images=None, mode=None, filename=None):
        self._batchsize = bs
        self._index_in_epoch = 0
        self._num_examples = count
        self._images = images
        self._mode = mode
        self._filename = filename
        self._epochs = ep
        self._classes = cls

    # decoding tfrecords; return images and labels
    def decode(self, serialized_example):
        features = tf.parse_example(
            serialized_example,
            # Defaults are not specified since both keys are required.
            features={self._mode + '/imageL0': tf.FixedLenFeature([], tf.string),
                      self._mode + '/imageL1': tf.FixedLenFeature([], tf.string),
                      self._mode + '/imageL2': tf.FixedLenFeature([], tf.string),
                      self._mode + '/age': tf.FixedLenFeature([], tf.float32),
                      self._mode + '/BMI': tf.FixedLenFeature([], tf.float32),
                      self._mode + '/label': tf.FixedLenFeature([], tf.int64), })

        imagea = tf.decode_raw(features[self._mode + '/imageL0'], tf.float32)
        imagea = tf.reshape(imagea, [-1, 299, 299, 3])
        imageb = tf.decode_raw(features[self._mode + '/imageL1'], tf.float32)
        imageb = tf.reshape(imageb, [-1, 299, 299, 3])
        imagec = tf.decode_raw(features[self._mode + '/imageL2'], tf.float32)
        imagec = tf.reshape(imagec, [-1, 299, 299, 3])

        age = tf.cast(features[self._mode + '/age'], tf.float32)
        BMI = tf.cast(features[self._mode + '/BMI'], tf.float32)
        demographic = tf.concat([age, BMI], -1)

        # Convert label from a scalar uint8 tensor to an int32 scalar.
        label = tf.cast(features[self._mode + '/label'], tf.int32)
        return imagea, imageb, imagec, label, demographic

    # decoding tfrecords for real test
    def Real_decode(self, serialized_example):
        features = tf.parse_example(
            serialized_example,
            # Defaults are not specified since both keys are required.
            features={self._mode + '/imageL0': tf.FixedLenFeature([], tf.string),
                      self._mode + '/imageL1': tf.FixedLenFeature([], tf.string),
                      self._mode + '/imageL2': tf.FixedLenFeature([], tf.string),
                      self._mode + '/age': tf.FixedLenFeature([], tf.float32),
                      self._mode + '/BMI': tf.FixedLenFeature([], tf.float32), })

        imagea = tf.decode_raw(features[self._mode + '/imageL0'], tf.float32)
        imagea = tf.reshape(imagea, [-1, 299, 299, 3])
        imageb = tf.decode_raw(features[self._mode + '/imageL1'], tf.float32)
        imageb = tf.reshape(imageb, [-1, 299, 299, 3])
        imagec = tf.decode_raw(features[self._mode + '/imageL2'], tf.float32)
        imagec = tf.reshape(imagec, [-1, 299, 299, 3])

        age = tf.cast(features[self._mode + '/age'], tf.float32)
        BMI = tf.cast(features[self._mode + '/BMI'], tf.float32)
        demographic = tf.concat([age, BMI], -1)

        return imagea, imageb, imagec, demographic

    # augmentation including onehot encoding
    def augment(self, imagea, imageb, imagec, labels, demographics):

        angles = tf.cast(tf.random_uniform([], 0, 4), tf.int32)
        imagea = tf.image.rot90(imagea, k=angles)
        imagea = tf.image.random_flip_left_right(imagea)
        imagea = tf.image.random_flip_up_down(imagea)
        imagea = tf.image.random_hue(imagea, 0.02)
        imagea = tf.image.random_brightness(imagea, 0.02)
        imagea = tf.image.random_contrast(imagea, 0.9, 1.1)
        imagea = tf.image.random_saturation(imagea, 0.9, 1.1)

        imageb = tf.image.rot90(imageb, k=angles)
        imageb = tf.image.random_flip_left_right(imageb)
        imageb = tf.image.random_flip_up_down(imageb)
        imageb = tf.image.random_hue(imageb, 0.02)
        imageb = tf.image.random_brightness(imageb, 0.02)
        imageb = tf.image.random_contrast(imageb, 0.9, 1.1)
        imageb = tf.image.random_saturation(imageb, 0.9, 1.1)

        imagec = tf.image.rot90(imagec, k=angles)
        imagec = tf.image.random_flip_left_right(imagec)
        imagec = tf.image.random_flip_up_down(imagec)
        imagec = tf.image.random_hue(imagec, 0.02)
        imagec = tf.image.random_brightness(imagec, 0.02)
        imagec = tf.image.random_contrast(imagec, 0.9, 1.1)
        imagec = tf.image.random_saturation(imagec, 0.9, 1.1)

        labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=self._classes)

        return imagea, imageb, imagec, labels, demographics

    # onehot encoding only; for test set
    def onehot_only(self, imagea, imageb, imagec, labels, demographics):
        with tf.name_scope('onehot_only'):
            labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=self._classes)
        return imagea, imageb, imagec, labels, demographics

    # dataset preparation; batching; Real test or not; train or test
    def data(self, Not_Realtest=True, train=True):
        batch_size = self._batchsize
        ep = self._epochs
        filenames = tf.placeholder(tf.string, shape=None)
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.repeat(ep)
        if Not_Realtest:
            if train:
                batched_dataset = dataset.batch(batch_size, drop_remainder=True)
                batched_dataset = batched_dataset.map(self.decode)
                batched_dataset = batched_dataset.map(self.augment)
            else:
                batched_dataset = dataset.batch(batch_size, drop_remainder=False)
                batched_dataset = batched_dataset.map(self.decode)
                batched_dataset = batched_dataset.map(self.onehot_only)
        else:
            batched_dataset = dataset.batch(batch_size, drop_remainder=False)
            batched_dataset = batched_dataset.map(self.Real_decode)
        iterator = batched_dataset.make_initializable_iterator()
        return iterator, self._filename, filenames

    @property
    def images(self):
        return self._images

    @property
    def num_examples(self):
        return self._num_examples

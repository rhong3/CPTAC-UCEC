"""
Data input preparation from decoding TFrecords, onehot encoding, augmentation, and batching

Created on 11/01/2018

@author: RH
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import math


class DataSet(object):
    # bs is batch size; ep is epoch; images are images; mode is test/train; filename is tfrecords
    def __init__(self, bs, count, ep=1, images = None, mode = None, filename = None):
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
    def augment(self, images, labels,
                resize=None,  # (width, height) tuple or None
                horizontal_flip=True,
                vertical_flip=True,
                rotate=0,  # Maximum rotation angle in degrees
                crop_probability=0,  # How often we do crops
                crop_min_percent=0.6,  # Minimum linear dimension of a crop
                crop_max_percent=1.,  # Maximum linear dimension of a crop
                mixup=0.4):
        if resize is not None:
            images = tf.image.resize_bilinear(images, resize)

        # My experiments showed that casting on GPU improves training performance
        if images.dtype != tf.float32:
            images = tf.image.convert_image_dtype(images, dtype=tf.float32)
            images = tf.subtract(images, 0.5)
            images = tf.multiply(images, 2.0)
        labels = tf.to_float(labels)

        with tf.name_scope('augmentation'):
            shp = tf.shape(images)
            batch_size, height, width = shp[0], shp[1], shp[2]
            width = tf.cast(width, tf.float32)
            height = tf.cast(height, tf.float32)

            # The list of affine transformations that our image will go under.
            # Every element is Nx8 tensor, where N is a batch size.
            transforms = []
            identity = tf.constant([1, 0, 0, 0, 1, 0, 0, 0], dtype=tf.float32)
            if horizontal_flip:
                coin = tf.less(tf.random_uniform([batch_size], 0, 1.0), 0.5)
                flip_transform = tf.convert_to_tensor(
                    [-1., 0., width, 0., 1., 0., 0., 0.], dtype=tf.float32)
                transforms.append(
                    tf.where(coin,
                             tf.tile(tf.expand_dims(flip_transform, 0), [batch_size, 1]),
                             tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])))

            if vertical_flip:
                coin = tf.less(tf.random_uniform([batch_size], 0, 1.0), 0.5)
                flip_transform = tf.convert_to_tensor(
                    [1, 0, 0, 0, -1, height, 0, 0], dtype=tf.float32)
                transforms.append(
                    tf.where(coin,
                             tf.tile(tf.expand_dims(flip_transform, 0), [batch_size, 1]),
                             tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])))

            if rotate > 0:
                angle_rad = rotate / 180 * math.pi
                angles = tf.random_uniform([batch_size], -angle_rad, angle_rad)
                transforms.append(
                    tf.contrib.image.angles_to_projective_transforms(
                        angles, height, width))

            if crop_probability > 0:
                crop_pct = tf.random_uniform([batch_size], crop_min_percent,
                                             crop_max_percent)
                left = tf.random_uniform([batch_size], 0, width * (1 - crop_pct))
                top = tf.random_uniform([batch_size], 0, height * (1 - crop_pct))
                crop_transform = tf.stack([
                    crop_pct,
                    tf.zeros([batch_size]), top,
                    tf.zeros([batch_size]), crop_pct, left,
                    tf.zeros([batch_size]),
                    tf.zeros([batch_size])
                ], 1)

                coin = tf.less(
                    tf.random_uniform([batch_size], 0, 1.0), crop_probability)
                transforms.append(
                    tf.where(coin, crop_transform,
                             tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])))

            if transforms:
                images = tf.contrib.image.transform(
                    images,
                    tf.contrib.image.compose_transforms(*transforms),
                    interpolation='BILINEAR')  # or 'NEAREST'

            labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=4)

            if mixup > 0:
                # Mixup coeffecient, see https://arxiv.org/abs/1710.09412.pdf
                def cshift(values):  # Circular shift in batch dimension
                    return tf.concat([values[-1:, ...], values[:-1, ...]], 0)

                mixup = 1.0 * mixup  # Convert to float, as tf.distributions.Beta requires floats.
                beta = tf.distributions.Beta(mixup, mixup)
                lam = beta.sample(batch_size)
                ll = tf.expand_dims(tf.expand_dims(tf.expand_dims(lam, -1), -1), -1)
                ly = tf.tile(tf.expand_dims(lam, -1), [1, 4])
                images = ll * images + (1 - ll) * cshift(images)
                labels = ly * labels + (1 - ly) * cshift(labels)

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

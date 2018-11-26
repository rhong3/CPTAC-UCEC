from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class DataSet(object):

    def __init__(self, bs, count, ep=1, images = None, mode = None, filename = None):
        self._batchsize = bs
        self._index_in_epoch = 0
        self._num_examples = count
        self._images = images
        self._mode = mode
        self._filename = filename
        self._epochs = ep

    def decode(self, serialized_example):
        features = tf.parse_example(
            serialized_example,
            # Defaults are not specified since both keys are required.
            features={self._mode + '/image': tf.FixedLenFeature([], tf.string),
                      self._mode + '/label': tf.FixedLenFeature([], tf.int64), })

        image = tf.decode_raw(features[self._mode + '/image'], tf.float32)
        image = tf.reshape(image, [self._batchsize, 299, 299, 3])

        # Convert label from a scalar uint8 tensor to an int32 scalar.
        label = tf.cast(features[self._mode + '/label'], tf.int32)
        return image, label

    def data(self, Not_Realtest=True):
        batch_size = self._batchsize
        ep = self._epochs
        if Not_Realtest:
            filenames = tf.placeholder(tf.string, shape=None)
            dataset = tf.data.TFRecordDataset(filenames)
            dataset = dataset.repeat(ep)
            batched_dataset = dataset.batch(batch_size, drop_remainder=True)
            batched_dataset = batched_dataset.map(self.decode)
            iterator = batched_dataset.make_initializable_iterator()
            # next_element = iterator.get_next()
            # with tf.Session() as sess:
            #     sess.run(iterator.initializer, feed_dict={filenames: self._filename})
            #     batch = sess.run(next_element)
            return iterator, self._filename, filenames
        else:
            features_placeholder = tf.placeholder(self._images.dtype, self._images.shape)
            dataset = tf.data.Dataset.from_tensor_slices(features_placeholder)
            dataset = dataset.repeat(ep)
            batched_dataset = dataset.batch(batch_size, drop_remainder=False)
            iterator = batched_dataset.make_initializable_iterator()
            # next_element = iterator.get_next()
            # with tf.Session() as sess:
            #     sess.run(iterator.initializer, feed_dict={features_placeholder: self._images})
            #     batch = sess.run(next_element)
            return iterator, self._images, features_placeholder

    @property
    def images(self):
        return self._images

    @property
    def num_examples(self):
        return self._num_examples

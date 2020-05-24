"""
InceptionV1 (GoogLeNet) for TF2.0

Created on 03/18/2019

@author: RH
"""

import tensorflow as tf
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import concatenate
from keras.regularizers import l2


def googlenet(input,
              dropout_keep_prob=0.8,
              num_classes=1000,
              is_training=True,
              scope='GoogleNet'):
    # creates GoogLeNet a.k.a. Inception v1 (Szegedy, 2015)
    with tf.name_scope(scope, "googlenet", [input]):

        conv1_7x7_s2 = Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu', name='conv1/7x7_s2',
                                     kernel_regularizer=l2(0.0002))(input)

        conv1_zero_pad = ZeroPadding2D(padding=(1, 1))(conv1_7x7_s2)

        pool1_3x3_s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name='pool1/3x3_s2')(
            conv1_zero_pad)

        pool1_norm1 = BatchNormalization(axis=3, scale=False, name='pool1/norm1')(pool1_3x3_s2)

        conv2_3x3_reduce = Conv2D(64, (1, 1), padding='same', activation='relu', name='conv2/3x3_reduce',
                                         kernel_regularizer=l2(0.0002))(pool1_norm1)

        conv2_3x3 = Conv2D(192, (3, 3), padding='same', activation='relu', name='conv2/3x3',
                                  kernel_regularizer=l2(0.0002))(conv2_3x3_reduce)

        conv2_norm2 = BatchNormalization(axis=3, scale=False, name='conv2/norm2')(conv2_3x3)

        conv2_zero_pad = ZeroPadding2D(padding=(1, 1))(conv2_norm2)

        pool2_3x3_s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name='pool2/3x3_s2')(
            conv2_zero_pad)

        inception_3a_1x1 = Conv2D(64, (1, 1), padding='same', activation='relu', name='inception_3a/1x1',
                                         kernel_regularizer=l2(0.0002))(pool2_3x3_s2)

        inception_3a_3x3_reduce = Conv2D(96, (1, 1), padding='same', activation='relu',
                                                name='inception_3a/3x3_reduce', kernel_regularizer=l2(0.0002))(pool2_3x3_s2)

        inception_3a_3x3 = Conv2D(128, (3, 3), padding='same', activation='relu', name='inception_3a/3x3',
                                         kernel_regularizer=l2(0.0002))(inception_3a_3x3_reduce)

        inception_3a_5x5_reduce = Conv2D(16, (1, 1), padding='same', activation='relu',
                                                name='inception_3a/5x5_reduce', kernel_regularizer=l2(0.0002))(pool2_3x3_s2)

        inception_3a_5x5 = Conv2D(32, (5, 5), padding='same', activation='relu', name='inception_3a/5x5',
                                         kernel_regularizer=l2(0.0002))(inception_3a_5x5_reduce)

        inception_3a_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='inception_3a/pool')(
            pool2_3x3_s2)

        inception_3a_pool_proj = Conv2D(32, (1, 1), padding='same', activation='relu',
                                               name='inception_3a/pool_proj', kernel_regularizer=l2(0.0002))(inception_3a_pool)

        inception_3a_output = concatenate([inception_3a_1x1, inception_3a_3x3, inception_3a_5x5, inception_3a_pool_proj],
                                          axis=3, name='inception_3a/output')

        inception_3b_1x1 = Conv2D(128, (1, 1), padding='same', activation='relu', name='inception_3b/1x1',
                                         kernel_regularizer=l2(0.0002))(inception_3a_output)

        inception_3b_3x3_reduce = Conv2D(128, (1, 1), padding='same', activation='relu',
                                                name='inception_3b/3x3_reduce', kernel_regularizer=l2(0.0002))(
            inception_3a_output)

        inception_3b_3x3 = Conv2D(192, (3, 3), padding='same', activation='relu', name='inception_3b/3x3',
                                         kernel_regularizer=l2(0.0002))(inception_3b_3x3_reduce)

        inception_3b_5x5_reduce = Conv2D(32, (1, 1), padding='same', activation='relu',
                                                name='inception_3b/5x5_reduce', kernel_regularizer=l2(0.0002))(
            inception_3a_output)

        inception_3b_5x5 = Conv2D(96, (5, 5), padding='same', activation='relu', name='inception_3b/5x5',
                                         kernel_regularizer=l2(0.0002))(inception_3b_5x5_reduce)

        inception_3b_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='inception_3b/pool')(
            inception_3a_output)

        inception_3b_pool_proj = Conv2D(64, (1, 1), padding='same', activation='relu',
                                               name='inception_3b/pool_proj', kernel_regularizer=l2(0.0002))(inception_3b_pool)

        inception_3b_output = concatenate([inception_3b_1x1, inception_3b_3x3, inception_3b_5x5, inception_3b_pool_proj],
                                          axis=3, name='inception_3b/output')

        inception_3b_output_zero_pad = ZeroPadding2D(padding=(1, 1))(inception_3b_output)

        pool3_3x3_s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name='pool3/3x3_s2')(
            inception_3b_output_zero_pad)

        inception_4a_1x1 = Conv2D(192, (1, 1), padding='same', activation='relu', name='inception_4a/1x1',
                                         kernel_regularizer=l2(0.0002))(pool3_3x3_s2)

        inception_4a_3x3_reduce = Conv2D(96, (1, 1), padding='same', activation='relu',
                                                name='inception_4a/3x3_reduce', kernel_regularizer=l2(0.0002))(pool3_3x3_s2)

        inception_4a_3x3 = Conv2D(208, (3, 3), padding='same', activation='relu', name='inception_4a/3x3',
                                         kernel_regularizer=l2(0.0002))(inception_4a_3x3_reduce)

        inception_4a_5x5_reduce = Conv2D(16, (1, 1), padding='same', activation='relu',
                                                name='inception_4a/5x5_reduce', kernel_regularizer=l2(0.0002))(pool3_3x3_s2)

        inception_4a_5x5 = Conv2D(48, (5, 5), padding='same', activation='relu', name='inception_4a/5x5',
                                         kernel_regularizer=l2(0.0002))(inception_4a_5x5_reduce)

        inception_4a_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='inception_4a/pool')(
            pool3_3x3_s2)

        inception_4a_pool_proj = Conv2D(64, (1, 1), padding='same', activation='relu',
                                               name='inception_4a/pool_proj', kernel_regularizer=l2(0.0002))(inception_4a_pool)

        inception_4a_output = concatenate([inception_4a_1x1, inception_4a_3x3, inception_4a_5x5, inception_4a_pool_proj],
                                          axis=3, name='inception_4a/output')

        loss1_ave_pool = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), name='loss1/ave_pool')(inception_4a_output)

        loss1_conv = Conv2D(128, (1, 1), padding='same', activation='relu', name='loss1/conv',
                                   kernel_regularizer=l2(0.0002))(loss1_ave_pool)

        loss1_flat = Flatten()(loss1_conv)

        loss1_fc = Dense(1024, activation='relu', name='loss1/fc', kernel_regularizer=l2(0.0002))(loss1_flat)

        loss1_drop_fc = Dropout(rate=dropout_keep_prob)(loss1_fc, training=is_training)

        loss1_classifier = Dense(num_classes, name='loss1/classifier', kernel_regularizer=l2(0.0002))(loss1_drop_fc)

        inception_4b_1x1 = Conv2D(160, (1, 1), padding='same', activation='relu', name='inception_4b/1x1',
                                         kernel_regularizer=l2(0.0002))(inception_4a_output)

        inception_4b_3x3_reduce = Conv2D(112, (1, 1), padding='same', activation='relu',
                                                name='inception_4b/3x3_reduce', kernel_regularizer=l2(0.0002))(
            inception_4a_output)

        inception_4b_3x3 = Conv2D(224, (3, 3), padding='same', activation='relu', name='inception_4b/3x3',
                                         kernel_regularizer=l2(0.0002))(inception_4b_3x3_reduce)

        inception_4b_5x5_reduce = Conv2D(24, (1, 1), padding='same', activation='relu',
                                                name='inception_4b/5x5_reduce', kernel_regularizer=l2(0.0002))(
            inception_4a_output)

        inception_4b_5x5 = Conv2D(64, (5, 5), padding='same', activation='relu', name='inception_4b/5x5',
                                         kernel_regularizer=l2(0.0002))(inception_4b_5x5_reduce)

        inception_4b_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='inception_4b/pool')(
            inception_4a_output)

        inception_4b_pool_proj = Conv2D(64, (1, 1), padding='same', activation='relu',
                                               name='inception_4b/pool_proj', kernel_regularizer=l2(0.0002))(inception_4b_pool)

        inception_4b_output = concatenate([inception_4b_1x1, inception_4b_3x3, inception_4b_5x5, inception_4b_pool_proj],
                                          axis=3, name='inception_4b_output')

        inception_4c_1x1 = Conv2D(128, (1, 1), padding='same', activation='relu', name='inception_4c/1x1',
                                         kernel_regularizer=l2(0.0002))(inception_4b_output)

        inception_4c_3x3_reduce = Conv2D(128, (1, 1), padding='same', activation='relu',
                                                name='inception_4c/3x3_reduce', kernel_regularizer=l2(0.0002))(
            inception_4b_output)

        inception_4c_3x3 = Conv2D(256, (3, 3), padding='same', activation='relu', name='inception_4c/3x3',
                                         kernel_regularizer=l2(0.0002))(inception_4c_3x3_reduce)

        inception_4c_5x5_reduce = Conv2D(24, (1, 1), padding='same', activation='relu',
                                                name='inception_4c/5x5_reduce', kernel_regularizer=l2(0.0002))(
            inception_4b_output)

        inception_4c_5x5 = Conv2D(64, (5, 5), padding='same', activation='relu', name='inception_4c/5x5',
                                         kernel_regularizer=l2(0.0002))(inception_4c_5x5_reduce)

        inception_4c_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='inception_4c/pool')(
            inception_4b_output)

        inception_4c_pool_proj = Conv2D(64, (1, 1), padding='same', activation='relu',
                                               name='inception_4c/pool_proj', kernel_regularizer=l2(0.0002))(inception_4c_pool)

        inception_4c_output = concatenate([inception_4c_1x1, inception_4c_3x3, inception_4c_5x5, inception_4c_pool_proj],
                                          axis=3, name='inception_4c/output')

        inception_4d_1x1 = Conv2D(112, (1, 1), padding='same', activation='relu', name='inception_4d/1x1',
                                         kernel_regularizer=l2(0.0002))(inception_4c_output)

        inception_4d_3x3_reduce = Conv2D(144, (1, 1), padding='same', activation='relu',
                                                name='inception_4d/3x3_reduce', kernel_regularizer=l2(0.0002))(
            inception_4c_output)

        inception_4d_3x3 = Conv2D(288, (3, 3), padding='same', activation='relu', name='inception_4d/3x3',
                                         kernel_regularizer=l2(0.0002))(inception_4d_3x3_reduce)

        inception_4d_5x5_reduce = Conv2D(32, (1, 1), padding='same', activation='relu',
                                                name='inception_4d/5x5_reduce', kernel_regularizer=l2(0.0002))(
            inception_4c_output)

        inception_4d_5x5 = Conv2D(64, (5, 5), padding='same', activation='relu', name='inception_4d/5x5',
                                         kernel_regularizer=l2(0.0002))(inception_4d_5x5_reduce)

        inception_4d_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='inception_4d/pool')(
            inception_4c_output)

        inception_4d_pool_proj = Conv2D(64, (1, 1), padding='same', activation='relu',
                                               name='inception_4d/pool_proj', kernel_regularizer=l2(0.0002))(inception_4d_pool)

        inception_4d_output = concatenate([inception_4d_1x1, inception_4d_3x3, inception_4d_5x5, inception_4d_pool_proj],
                                          axis=3, name='inception_4d/output')

        loss2_ave_pool = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), name='loss2/ave_pool')(inception_4d_output)

        loss2_conv = Conv2D(128, (1, 1), padding='same', activation='relu', name='loss2/conv',
                                   kernel_regularizer=l2(0.0002))(loss2_ave_pool)

        loss2_flat = Flatten()(loss2_conv)

        loss2_fc = Dense(1024, activation='relu', name='loss2/fc', kernel_regularizer=l2(0.0002))(loss2_flat)

        loss2_drop_fc = Dropout(rate=dropout_keep_prob)(loss2_fc, training=is_training)

        loss2_classifier = Dense(num_classes, name='loss2/classifier', kernel_regularizer=l2(0.0002))(loss2_drop_fc)

        inception_4e_1x1 = Conv2D(256, (1, 1), padding='same', activation='relu', name='inception_4e/1x1',
                                         kernel_regularizer=l2(0.0002))(inception_4d_output)

        inception_4e_3x3_reduce = Conv2D(160, (1, 1), padding='same', activation='relu',
                                                name='inception_4e/3x3_reduce', kernel_regularizer=l2(0.0002))(
            inception_4d_output)

        inception_4e_3x3 = Conv2D(320, (3, 3), padding='same', activation='relu', name='inception_4e/3x3',
                                         kernel_regularizer=l2(0.0002))(inception_4e_3x3_reduce)

        inception_4e_5x5_reduce = Conv2D(32, (1, 1), padding='same', activation='relu',
                                                name='inception_4e/5x5_reduce', kernel_regularizer=l2(0.0002))(
            inception_4d_output)

        inception_4e_5x5 = Conv2D(128, (5, 5), padding='same', activation='relu', name='inception_4e/5x5',
                                         kernel_regularizer=l2(0.0002))(inception_4e_5x5_reduce)

        inception_4e_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='inception_4e/pool')(
            inception_4d_output)

        inception_4e_pool_proj = Conv2D(128, (1, 1), padding='same', activation='relu',
                                               name='inception_4e/pool_proj', kernel_regularizer=l2(0.0002))(inception_4e_pool)

        inception_4e_output = concatenate([inception_4e_1x1, inception_4e_3x3, inception_4e_5x5, inception_4e_pool_proj],
                                          axis=3, name='inception_4e/output')

        inception_4e_output_zero_pad = ZeroPadding2D(padding=(1, 1))(inception_4e_output)

        pool4_3x3_s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name='pool4/3x3_s2')(
            inception_4e_output_zero_pad)

        inception_5a_1x1 = Conv2D(256, (1, 1), padding='same', activation='relu', name='inception_5a/1x1',
                                         kernel_regularizer=l2(0.0002))(pool4_3x3_s2)

        inception_5a_3x3_reduce = Conv2D(160, (1, 1), padding='same', activation='relu',
                                                name='inception_5a/3x3_reduce', kernel_regularizer=l2(0.0002))(pool4_3x3_s2)

        inception_5a_3x3 = Conv2D(320, (3, 3), padding='same', activation='relu', name='inception_5a/3x3',
                                         kernel_regularizer=l2(0.0002))(inception_5a_3x3_reduce)

        inception_5a_5x5_reduce = Conv2D(32, (1, 1), padding='same', activation='relu',
                                                name='inception_5a/5x5_reduce', kernel_regularizer=l2(0.0002))(pool4_3x3_s2)

        inception_5a_5x5 = Conv2D(128, (5, 5), padding='same', activation='relu', name='inception_5a/5x5',
                                         kernel_regularizer=l2(0.0002))(inception_5a_5x5_reduce)

        inception_5a_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='inception_5a/pool')(
            pool4_3x3_s2)

        inception_5a_pool_proj = Conv2D(128, (1, 1), padding='same', activation='relu',
                                               name='inception_5a/pool_proj', kernel_regularizer=l2(0.0002))(inception_5a_pool)

        inception_5a_output = concatenate([inception_5a_1x1, inception_5a_3x3, inception_5a_5x5, inception_5a_pool_proj],
                                          axis=3, name='inception_5a/output')

        inception_5b_1x1 = Conv2D(384, (1, 1), padding='same', activation='relu', name='inception_5b/1x1',
                                         kernel_regularizer=l2(0.0002))(inception_5a_output)

        inception_5b_3x3_reduce = Conv2D(192, (1, 1), padding='same', activation='relu',
                                                name='inception_5b/3x3_reduce', kernel_regularizer=l2(0.0002))(
            inception_5a_output)

        inception_5b_3x3 = Conv2D(384, (3, 3), padding='same', activation='relu', name='inception_5b/3x3',
                                         kernel_regularizer=l2(0.0002))(inception_5b_3x3_reduce)

        inception_5b_5x5_reduce = Conv2D(48, (1, 1), padding='same', activation='relu',
                                                name='inception_5b/5x5_reduce', kernel_regularizer=l2(0.0002))(
            inception_5a_output)

        inception_5b_5x5 = Conv2D(128, (5, 5), padding='same', activation='relu', name='inception_5b/5x5',
                                         kernel_regularizer=l2(0.0002))(inception_5b_5x5_reduce)

        inception_5b_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='inception_5b/pool')(
            inception_5a_output)

        inception_5b_pool_proj = Conv2D(128, (1, 1), padding='same', activation='relu',
                                               name='inception_5b/pool_proj', kernel_regularizer=l2(0.0002))(inception_5b_pool)

        inception_5b_output = concatenate([inception_5b_1x1, inception_5b_3x3, inception_5b_5x5, inception_5b_pool_proj],
                                          axis=3, name='inception_5b/output')

        net = inception_5b_output

        # Modified for 299x299
        pool5_10x10_s1 = AveragePooling2D(pool_size=(10, 10), strides=(1, 1), name='pool5/10x10_s2')(inception_5b_output)

        loss3_flat = Flatten()(pool5_10x10_s1)

        pool5_drop_10x10_s1 = Dropout(rate=dropout_keep_prob)(loss3_flat, training=is_training)

        loss3_classifier_w = Dense(num_classes, name='loss3/classifier', kernel_regularizer=l2(0.0002))

        loss3_classifier = loss3_classifier_w(pool5_drop_10x10_s1)

        w_variables = loss3_classifier_w.get_weights()

        logits = tf.cond(tf.equal(is_training, tf.constant(True)),
                         lambda: tf.add(loss3_classifier, tf.scalar_mul(tf.constant(0.3), tf.add(loss1_classifier, loss2_classifier))),
                         lambda: loss3_classifier)

    return logits, net, tf.convert_to_tensor(w_variables[0])


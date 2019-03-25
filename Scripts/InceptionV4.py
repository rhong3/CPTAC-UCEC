#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
InceptionV4 for TF2.0

Created on 03/19/2019

@author: RH
"""
import tensorflow as tf
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import concatenate
from keras.regularizers import l2


def Conv2D_n(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              name=None):

    if name is not None:
        conv_name = name + '_conv'
    else:
        conv_name = None
    x = Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name,
        data_format="channels_last",
        kernel_regularizer=l2(0.0002))(x)
    x = Activation('relu', name=name)(x)
    return x


def stem(input):
    '''The stem of the pure Inception-v4 and Inception-ResNet-v2 networks. This is input part of those networks.'''

    # Input shape is 299 * 299 * 3 (Tensorflow dimension ordering)
    x = Conv2D_n(input, 32, 3, 3, strides=(2, 2), padding="same")  # 149 * 149 * 32
    x = Conv2D_n(x, 32, 3, 3, padding="same")  # 147 * 147 * 32
    x = Conv2D_n(x, 64, 3, 3)  # 147 * 147 * 64

    x1 = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    x2 = Conv2D_n(x, 96, 3, 3, strides=(2, 2), padding="same")

    x = concatenate([x1, x2], axis=-1)  # 73 * 73 * 160

    x1 = Conv2D_n(x, 64, 1, 1)
    x1 = Conv2D_n(x1, 96, 3, 3, padding="same")

    x2 = Conv2D_n(x, 64, 1, 1)
    x2 = Conv2D_n(x2, 64, 1, 7)
    x2 = Conv2D_n(x2, 64, 7, 1)
    x2 = Conv2D_n(x2, 96, 3, 3, padding="same")

    x = concatenate([x1, x2], axis=-1)  # 71 * 71 * 192

    x1 = Conv2D_n(x, 192, 3, 3, strides=(2, 2), padding="same")

    x2 = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

    x = concatenate([x1, x2], axis=-1)  # 35 * 35 * 384

    x = BatchNormalization(axis=-1)(x)

    x = Activation('relu')(x)

    return x


def inception_A(input):
    '''Architecture of Inception_A block which is a 35 * 35 grid module.'''

    a1 = AveragePooling2D((3, 3), strides=(1, 1), padding="same")(input)
    a1 = Conv2D_n(a1, 96, 1, 1)

    a2 = Conv2D_n(input, 96, 1, 1)

    a3 = Conv2D_n(input, 64, 1, 1)
    a3 = Conv2D_n(a3, 96, 3, 3)

    a4 = Conv2D_n(input, 64, 1, 1)
    a4 = Conv2D_n(a4, 96, 3, 3)
    a4 = Conv2D_n(a4, 96, 3, 3)

    merged = concatenate([a1, a2, a3, a4], axis=-1)

    merged = BatchNormalization(axis=-1)(merged)

    merged = Activation('relu')(merged)

    return merged


def inception_B(input):
    '''Architecture of Inception_B block which is a 17 * 17 grid module.'''

    b1 = AveragePooling2D((3, 3), strides=(1, 1), padding="same")(input)
    b1 = Conv2D_n(b1, 128, 1, 1)

    b2 = Conv2D_n(input, 384, 1, 1)

    b3 = Conv2D_n(input, 192, 1, 1)
    b3 = Conv2D_n(b3, 224, 1, 7)
    b3 = Conv2D_n(b3, 256, 7, 1)

    b4 = Conv2D_n(input, 192, 1, 1)
    b4 = Conv2D_n(b4, 192, 7, 1)
    b4 = Conv2D_n(b4, 224, 1, 7)
    b4 = Conv2D_n(b4, 224, 7, 1)
    b4 = Conv2D_n(b4, 256, 1, 7)

    merged = concatenate([b1, b2, b3, b4], axis=-1)

    merged = BatchNormalization(axis=-1)(merged)

    merged = Activation('relu')(merged)

    return merged


def inception_C(input):
    '''Architecture of Inception_C block which is a 8 * 8 grid module.'''

    c1 = AveragePooling2D((3, 3), strides=(1, 1), padding="same")(input)
    c1 = Conv2D_n(c1, 256, 1, 1)

    c2 = Conv2D_n(input, 256, 1, 1)

    c3 = Conv2D_n(input, 384, 1, 1)
    c31 = Conv2D_n(c3, 256, 1, 3)
    c32 = Conv2D_n(c3, 256, 3, 1)
    c3 = concatenate([c31, c32], axis=-1)

    c4 = Conv2D_n(input, 384, 1, 1)
    c4 = Conv2D_n(c4, 448, 3, 1)
    c4 = Conv2D_n(c4, 512, 1, 3)
    c41 = Conv2D_n(c4, 256, 1, 3)
    c42 = Conv2D_n(c4, 256, 3, 1)
    c4 = concatenate([c41, c42], axis=-1)

    merged = concatenate([c1, c2, c3, c4], axis=-1)

    merged = BatchNormalization(axis=-1)(merged)

    merged = Activation('relu')(merged)

    return merged


def reduction_A(input, k=192, l=224, m=256, n=384):
    '''Architecture of a 35 * 35 to 17 * 17 Reduction_A block.'''

    ra1 = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(input)

    ra2 = Conv2D_n(input, n, 3, 3, strides=(2, 2), padding="same")

    ra3 = Conv2D_n(input, k, 1, 1)
    ra3 = Conv2D_n(ra3, l, 3, 3)
    ra3 = Conv2D_n(ra3, m, 3, 3, strides=(2, 2), padding="same")

    merged = concatenate([ra1, ra2, ra3], axis=-1)

    merged = BatchNormalization(axis=-1)(merged)

    merged = Activation('relu')(merged)

    return merged


def reduction_B(input):
    '''Architecture of a 17 * 17 to 8 * 8 Reduction_B block.'''

    rb1 = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(input)

    rb2 = Conv2D_n(input, 192, 1, 1)
    rb2 = Conv2D_n(rb2, 192, 3, 3, strides=(2, 2), padding="same")

    rb3 = Conv2D_n(input, 256, 1, 1)
    rb3 = Conv2D_n(rb3, 256, 1, 7)
    rb3 = Conv2D_n(rb3, 320, 7, 1)
    rb3 = Conv2D_n(rb3, 320, 3, 3, strides=(2, 2), padding="same")

    merged = concatenate([rb1, rb2, rb3], axis=-1)

    merged = BatchNormalization(axis=-1)(merged)

    merged = Activation('relu')(merged)

    return merged


def inceptionv4(input, dropout_keep_prob=0.8, num_classes=1000, is_training=True, scope='InceptionV4'):
    '''Creates the Inception_v4 network.'''
    with tf.name_scope(scope, "InceptionV4", [input]):
        # Input shape is 299 * 299 * 3
        x = stem(input)  # Output: 35 * 35 * 384

        # 4 x Inception A
        for i in range(4):
            x = inception_A(x)
            # Output: 35 * 35 * 384

        # Reduction A
        x = reduction_A(x, k=192, l=224, m=256, n=384)  # Output: 17 * 17 * 1024

        # 7 x Inception B
        for i in range(7):
            x = inception_B(x)
            # Output: 17 * 17 * 1024

        #Auxiliary
        loss2_ave_pool = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), name='loss2/ave_pool')(x)

        loss2_conv_a = Conv2D_n(loss2_ave_pool, 128, 1, 1)
        loss2_conv_b = Conv2D_n(loss2_conv_a, 768, 5, 5)

        loss2_conv_b = BatchNormalization(axis=-1)(loss2_conv_b)

        loss2_conv_b = Activation('relu')(loss2_conv_b)

        loss2_flat = Flatten()(loss2_conv_b)

        loss2_fc = Dense(1024, activation='relu', name='loss2/fc', kernel_regularizer=l2(0.0002))(loss2_flat)

        loss2_drop_fc = Dropout(dropout_keep_prob)(loss2_fc, training=is_training)

        loss2_classifier = Dense(num_classes, name='loss2/classifier', kernel_regularizer=l2(0.0002))(loss2_drop_fc)

        # Reduction B
        x = reduction_B(x)  # Output: 8 * 8 * 1536

        # 3 x Inception C
        for i in range(3):
            x = inception_C(x)
            # Output: 8 * 8 * 1536

        net = x

        # Average Pooling
        x = GlobalAveragePooling2D(name='avg_pool')(x)  # Output: 1536

        pool5_drop_10x10_s1 = Dropout(dropout_keep_prob)(x, training=is_training)

        loss3_classifier_W = Dense(num_classes, name='loss3/classifier', kernel_regularizer=l2(0.0002))

        loss3_classifier = loss3_classifier_W(pool5_drop_10x10_s1)

        w_variables = loss3_classifier_W

        logits = tf.math.add(loss3_classifier, tf.scalar_mul(tf.constant(0.3), loss2_classifier))

    return logits, net, w_variables


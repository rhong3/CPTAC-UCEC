"""
Resnet Code for TF2.0

Created on 11/04/2019

@author: RH
"""
import tensorflow as tf
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import add
from keras.regularizers import l2


def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              name=None):

    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    bn_axis = 3
    x = Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name,
        data_format="channels_last",
        kernel_regularizer=l2(0.0002))(x)
    x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x


def conv2d_bn_a(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              name=None):

    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    bn_axis = 3
    x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    x = Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name,
        data_format="channels_last",
        kernel_regularizer=l2(0.0002))(x)

    return x



def resnet_stem(input):
    x = conv2d_bn(input, 64, 7, 7, padding='valid')
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    return x


def block_1(input, sizea, sizeb, scale=0.5):
    input = Conv2D(
        sizeb, (1, 1),
        strides=(1, 1),
        padding='same',
        use_bias=False,
        data_format="channels_last",
        kernel_regularizer=l2(0.0002))(input)
    x = conv2d_bn_a(input, sizea, 3, 3)
    x = conv2d_bn_a(x, sizeb, 3, 3)
    x = x * scale
    output = add([input, x])
    return output


def block_2(input, sizea, sizeb, scale=0.5):
    input = Conv2D(
        sizeb, (1, 1),
        strides=(1, 1),
        padding='same',
        use_bias=False,
        data_format="channels_last",
        kernel_regularizer=l2(0.0002))(input)
    x = conv2d_bn_a(input, sizea, 1, 1)
    x = conv2d_bn_a(x, sizea, 3, 3)
    x = conv2d_bn_a(x, sizeb, 1, 1)
    x = x * scale
    output = add([input, x])
    return output


def resnet(input, mode=18, dropout_keep_prob=0.8, num_classes=1000, is_training=True, scope='ResNet'):
    if mode == 34:
        block = block_1
        repeats = [[3, 64, 64], [4, 128, 128], [6, 256, 256], [3, 512, 512]]
    elif mode == 50:
        block = block_2
        repeats = [[3, 64, 256], [4, 128, 512], [6, 256, 1024], [3, 512, 2048]]
    elif mode == 101:
        block = block_2
        repeats = [[3, 64, 256], [4, 128, 512], [23, 256, 1024], [3, 512, 2048]]
    elif mode == 152:
        block = block_2
        repeats = [[3, 64, 256], [8, 128, 512], [36, 256, 1024], [3, 512, 2048]]
    else:
        block = block_1
        repeats = [[2, 64, 64], [2, 128, 128], [2, 256, 256], [2, 512, 512]]

    x = resnet_stem(input)
    for m in repeats:
        for i in range(m[0]):
            x = block(x, m[1], m[2])

    net = x

    x = GlobalAveragePooling2D(name='avg_pool')(x)

    pool5_drop_10x10_s1 = Dropout(dropout_keep_prob)(x, training=is_training)

    loss3_classifier_w = Dense(num_classes, name='loss3/classifier', kernel_regularizer=l2(0.0002))

    loss3_classifier = loss3_classifier_w(pool5_drop_10x10_s1)

    w_variables = loss3_classifier_w.get_weights()

    return loss3_classifier, net, tf.convert_to_tensor(w_variables[0])


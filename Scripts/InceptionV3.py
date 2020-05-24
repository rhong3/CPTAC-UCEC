"""
InceptionV3 for TF2.0

Created on 03/18/2019

@author: RH
"""
import tensorflow as tf
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import concatenate
from keras.regularizers import l2


def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              train=True,
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
    x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x, training=train)
    x = Activation('relu', name=name)(x)
    return x


def inceptionv3(input, dropout_keep_prob=0.8, num_classes=1000, is_training=True,
                scope='InceptionV3', channel_axis=3):

    with tf.name_scope(scope, "InceptionV3", [input]):

        x = conv2d_bn(input, 32, 3, 3, strides=(2, 2), padding='valid')
        x = conv2d_bn(x, 32, 3, 3, padding='valid')
        x = conv2d_bn(x, 64, 3, 3)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = conv2d_bn(x, 80, 1, 1, padding='valid')
        x = conv2d_bn(x, 192, 3, 3, padding='valid')
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        # mixed 0: 35 x 35 x 256
        branch1x1 = conv2d_bn(x, 64, 1, 1)

        branch5x5 = conv2d_bn(x, 48, 1, 1)
        branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

        branch3x3dbl = conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

        branch_pool = AveragePooling2D((3, 3),
                                              strides=(1, 1),
                                              padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 32, 1, 1)
        x = concatenate(
            [branch1x1, branch5x5, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed0')

        # mixed 1: 35 x 35 x 288
        branch1x1 = conv2d_bn(x, 64, 1, 1)

        branch5x5 = conv2d_bn(x, 48, 1, 1)
        branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

        branch3x3dbl = conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

        branch_pool = AveragePooling2D((3, 3),
                                              strides=(1, 1),
                                              padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
        x = concatenate(
            [branch1x1, branch5x5, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed1')

        # mixed 2: 35 x 35 x 288
        branch1x1 = conv2d_bn(x, 64, 1, 1)

        branch5x5 = conv2d_bn(x, 48, 1, 1)
        branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

        branch3x3dbl = conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

        branch_pool = AveragePooling2D((3, 3),
                                              strides=(1, 1),
                                              padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
        x = concatenate(
            [branch1x1, branch5x5, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed2')

        # mixed 3: 17 x 17 x 768
        branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')

        branch3x3dbl = conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = conv2d_bn(
            branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')

        branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = concatenate(
            [branch3x3, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed3')

        # mixed 4: 17 x 17 x 768
        branch1x1 = conv2d_bn(x, 192, 1, 1)

        branch7x7 = conv2d_bn(x, 128, 1, 1)
        branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)
        branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = conv2d_bn(x, 128, 1, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = AveragePooling2D((3, 3),
                                              strides=(1, 1),
                                              padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=channel_axis,
            name='mixed4')

        # mixed 5, 6: 17 x 17 x 768
        for i in range(2):
            branch1x1 = conv2d_bn(x, 192, 1, 1)

            branch7x7 = conv2d_bn(x, 160, 1, 1)
            branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)
            branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

            branch7x7dbl = conv2d_bn(x, 160, 1, 1)
            branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
            branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7)
            branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
            branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

            branch_pool = AveragePooling2D(
                (3, 3), strides=(1, 1), padding='same')(x)
            branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
            x = concatenate(
                [branch1x1, branch7x7, branch7x7dbl, branch_pool],
                axis=channel_axis,
                name='mixed' + str(5 + i))

        # mixed 7: 17 x 17 x 768
        branch1x1 = conv2d_bn(x, 192, 1, 1)

        branch7x7 = conv2d_bn(x, 192, 1, 1)
        branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)
        branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = conv2d_bn(x, 192, 1, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = AveragePooling2D((3, 3),
                                              strides=(1, 1),
                                              padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=channel_axis,
            name='mixed7')

        loss2_ave_pool = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), name='loss2/ave_pool')(x)

        loss2_conv_a = conv2d_bn(loss2_ave_pool, 128, 1, 1)
        loss2_conv_b = conv2d_bn(loss2_conv_a, 768, 5, 5)

        loss2_flat = Flatten()(loss2_conv_b)

        loss2_fc = Dense(1024, activation='relu', name='loss2/fc', kernel_regularizer=l2(0.0002))(loss2_flat)

        loss2_drop_fc = Dropout(dropout_keep_prob)(loss2_fc, training=is_training)

        loss2_classifier = Dense(num_classes, name='loss2/classifier', kernel_regularizer=l2(0.0002))(loss2_drop_fc)

        # mixed 8: 8 x 8 x 1280
        branch3x3 = conv2d_bn(x, 192, 1, 1)
        branch3x3 = conv2d_bn(branch3x3, 320, 3, 3,
                              strides=(2, 2), padding='valid')

        branch7x7x3 = conv2d_bn(x, 192, 1, 1)
        branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
        branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
        branch7x7x3 = conv2d_bn(
            branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

        branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = concatenate(
            [branch3x3, branch7x7x3, branch_pool],
            axis=channel_axis,
            name='mixed8')

        # mixed 9: 8 x 8 x 2048
        for i in range(2):
            branch1x1 = conv2d_bn(x, 320, 1, 1)

            branch3x3 = conv2d_bn(x, 384, 1, 1)
            branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
            branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
            branch3x3 = concatenate(
                [branch3x3_1, branch3x3_2],
                axis=channel_axis,
                name='mixed9_' + str(i))

            branch3x3dbl = conv2d_bn(x, 448, 1, 1)
            branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
            branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
            branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
            branch3x3dbl = concatenate(
                [branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)

            branch_pool = AveragePooling2D(
                (3, 3), strides=(1, 1), padding='same')(x)
            branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
            x = concatenate(
                [branch1x1, branch3x3, branch3x3dbl, branch_pool],
                axis=channel_axis,
                name='mixed' + str(9 + i))
        net = x

        # Classification block
        x = GlobalAveragePooling2D(name='avg_pool')(x)

        pool5_drop_10x10_s1 = Dropout(dropout_keep_prob)(x, training=is_training)

        loss3_classifier_w = Dense(num_classes, name='loss3/classifier', kernel_regularizer=l2(0.0002))

        loss3_classifier = loss3_classifier_w(pool5_drop_10x10_s1)

        w_variables = loss3_classifier_w.get_weights()

        logits = tf.cond(tf.equal(is_training, tf.constant(True)),
                         lambda: tf.add(loss3_classifier, tf.scalar_mul(tf.constant(0.3), loss2_classifier)),
                         lambda: loss3_classifier)

    return logits, net, tf.convert_to_tensor(w_variables[0])

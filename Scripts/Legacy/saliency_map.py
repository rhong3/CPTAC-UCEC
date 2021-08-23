"""
Integrated gradient saliency maps

Created on 04/30/2020

@author: RH
"""

import saliency
import os
import sys
import cv2
import numpy as np
import tensorflow as tf
import data_input2 as data_input


# image to double
def im2double(im):
    return cv2.normalize(im.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)


# image to jpg
def py_map2jpg(imgmap):
    heatmap_x = np.round(imgmap*255).astype(np.uint8)
    return cv2.applyColorMap(heatmap_x, cv2.COLORMAP_JET)


def infer(model, classes, dropout=0.3):
    # image input
    x_in = tf.placeholder(tf.float32, name="x")
    x_in_reshape = tf.reshape(x_in, [-1, 299, 299, 3])

    # label input
    y_in = tf.placeholder(dtype=tf.int32, name="y")

    if model == 'I1':
        import InceptionV1
        logits, nett, ww = InceptionV1.googlenet(x_in_reshape,
                                                 num_classes=classes,
                                                 is_training=False,
                                                 dropout_keep_prob=dropout,
                                                 scope='GoogleNet')
        print('Using Inception-V1')
    elif model == 'I2':
        import InceptionV2
        logits, nett, ww = InceptionV2.inceptionv2(x_in_reshape,
                                                   num_classes=classes,
                                                   is_training=False,
                                                   dropout_keep_prob=dropout,
                                                   scope='InceptionV2')
        print('Using Inception-V2')
    elif model == 'I3':
        import InceptionV3
        logits, nett, ww = InceptionV3.inceptionv3(x_in_reshape,
                                                   num_classes=classes,
                                                   is_training=False,
                                                   dropout_keep_prob=dropout,
                                                   scope='InceptionV3')
        print('Using Inception-V3')
    elif model == 'I4':
        import InceptionV4
        logits, nett, ww = InceptionV4.inceptionv4(x_in_reshape,
                                                   num_classes=classes,
                                                   is_training=False,
                                                   dropout_keep_prob=dropout,
                                                   scope='InceptionV4')
        print('Using Inception-V4')
    elif model == 'I5':
        import InceptionV5
        logits, nett, ww = InceptionV5.inceptionresnetv1(x_in_reshape,
                                                         num_classes=classes,
                                                         is_training=False,
                                                         dropout_keep_prob=dropout,
                                                         scope='InceptionResV1')
        print('Using Inception-Resnet-V1')
    elif model == 'I6':
        import InceptionV6
        logits, nett, ww = InceptionV6.inceptionresnetv2(x_in_reshape,
                                                         num_classes=classes,
                                                         is_training=False,
                                                         dropout_keep_prob=dropout,
                                                         scope='InceptionResV2')
        print('Using Inception-Resnet-V2')
    elif model == 'R18':
        from Scripts.Legacy import ResNet
        logits, nett, ww = ResNet.resnet(x_in_reshape,
                                         mode=18,
                                         num_classes=classes,
                                         is_training=False,
                                         dropout_keep_prob=dropout,
                                         scope='ResNet18')
        print('Using ResNet18')
    elif model == 'R34':
        from Scripts.Legacy import ResNet
        logits, nett, ww = ResNet.resnet(x_in_reshape,
                                         mode=34,
                                         num_classes=classes,
                                         is_training=False,
                                         dropout_keep_prob=dropout,
                                         scope='ResNet34')
        print('Using ResNet34')
    elif model == 'R50':
        from Scripts.Legacy import ResNet
        logits, nett, ww = ResNet.resnet(x_in_reshape,
                                         mode=50,
                                         num_classes=classes,
                                         is_training=False,
                                         dropout_keep_prob=dropout,
                                         scope='ResNet50')
        print('Using ResNet50')
    elif model == 'R101':
        from Scripts.Legacy import ResNet
        logits, nett, ww = ResNet.resnet(x_in_reshape,
                                         mode=101,
                                         num_classes=classes,
                                         is_training=False,
                                         dropout_keep_prob=dropout,
                                         scope='ResNet101')
        print('Using ResNet101')
    elif model == 'R152':
        from Scripts.Legacy import ResNet
        logits, nett, ww = ResNet.resnet(x_in_reshape,
                                         mode=152,
                                         num_classes=classes,
                                         is_training=False,
                                         dropout_keep_prob=dropout,
                                         scope='ResNet152')
        print('Using ResNet152')
    else:
        import InceptionV1
        logits, nett, ww = InceptionV1.googlenet(x_in_reshape,
                                                 num_classes=classes,
                                                 is_training=False,
                                                 dropout_keep_prob=dropout,
                                                 scope='GoogleNet')
        print('Using Default: Inception-V1')

    pred = tf.nn.softmax(logits, name="prediction")

    neuron_selector = tf.placeholder(tf.int32)
    ny = logits[0][neuron_selector]

    return x_in, y_in, logits, nett, ww, pred, neuron_selector, ny


def reconstruct(X, model, classs, modelpath, outpath, do=0.3, bs =64):
    graph = tf.Graph()
    with graph.as_default():
        x_in_, y_in_, logits_, nett_, ww_, pred_, neuron_selector_, ny_ = infer(model=model, classes=classs, dropout=do)
        with tf.Session(graph=graph,
                        config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.import_meta_graph(str(modelpath+'.meta'))
            saver.restore(sess, modelpath)
            itr, file, ph = X.data(train=False)
            next_element = itr.get_next()
            with tf.Session() as sessa:
                sessa.run(itr.initializer, feed_dict={ph: file})
                ct = 0
                while True:
                    try:
                        x, y = sessa.run(next_element)
                        for mm in range(np.shape(x)[0]):
                            grad = saliency.IntegratedGradients(graph, sess, y, x_in_)
                            img = x[mm, :, :, :]
                            # Baseline is a white image.
                            baseline = np.zeros(img.shape)
                            baseline.fill(255)

                            smoothgrad_mask_3d = grad.GetSmoothedMask(x, feed_dict={
                                neuron_selector_: 1}, x_steps=25, x_baseline=baseline)

                            # Call the visualization methods to convert the 3D tensors to 2D grayscale.
                            smoothgrad_mask_grayscale = saliency.VisualizeImageGrayscale(smoothgrad_mask_3d)
                            smoothgrad_mask_grayscale = im2double(smoothgrad_mask_grayscale)
                            smoothgrad_mask_grayscale = py_map2jpg(smoothgrad_mask_grayscale)
                            sa = im2double(img) * 255
                            sb = im2double(smoothgrad_mask_grayscale) * 255
                            scurHeatMap = sa * 0.5 + sb * 0.5
                            sab = np.hstack((sa, sb))
                            sfull = np.hstack((scurHeatMap, sab))
                            cv2.imwrite(str(outpath + str(ct) + '.png'), sfull)

                            ct += 1
                    except tf.errors.OutOfRangeError:
                        print("Done!")
                        break


if __name__ == "__main__":
    THE = data_input.DataSet(64, 10000, ep=1, cls=2, mode='test', filename='PATH TO test.tfrecords')
    reconstruct(THE, 'I3', 2, 'PATH TO trained model', 'PATH TO output dir', do=0.3, bs=64)


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main method for Xeption

Created on 04/26/2019

@author: RH
"""
import os
import sys
import numpy as np
import tensorflow as tf
import data_input3
import cnn5
import pandas as pd
import cv2
import Sample_prep2
import time
import matplotlib
matplotlib.use('Agg')

dirr = sys.argv[1]  # output directory
bs = sys.argv[2]    # batch size
ep = sys.argv[3]    # epochs to train
md = sys.argv[4]    # architecture to use
pdmd = sys.argv[5]  # feature to predict


if pdmd == 'subtype':
    classes = 4
else:
    classes = 2

bs = int(bs)
ep = int(ep)

IMG_DIM = 299
# input image dimension
INPUT_DIM = [bs, IMG_DIM, IMG_DIM, 3]
# hyper parameters
HYPERPARAMS = {
    "batch_size": bs,
    "dropout": 0.3,
    "learning_rate": 1E-4,
    "classes": classes
}

MAX_ITER = np.inf
MAX_EPOCHS = ep
# paths to directories
img_dir = '../tiles/'
LOG_DIR = "../Results/{}".format(dirr)
METAGRAPH_DIR = "../Results/{}".format(dirr)
data_dir = "../Results/{}/data".format(dirr)
out_dir = "../Results/{}/out".format(dirr)


# count numbers of training and testing images
def counters(totlist_dir):
    trlist = pd.read_csv(totlist_dir + '/tr_sample.csv', header=0)
    telist = pd.read_csv(totlist_dir + '/te_sample.csv', header=0)
    valist = pd.read_csv(totlist_dir + '/va_sample.csv', header=0)
    trcc = len(trlist['label']) - 1
    tecc = len(telist['label']) - 1
    vacc = len(valist['label']) - 1

    return trcc, tecc, vacc


# read images
def load_image(addr):
    img = cv2.imread(addr)
    img = img.astype(np.float32)
    return img


# used for tfrecord labels generation
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# used for tfrecord images generation
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# loading images for dictionaries and generate tfrecords
def loader(totlist_dir):
    trlist = pd.read_csv(totlist_dir+'/tr_sample.csv', header=0)
    telist = pd.read_csv(totlist_dir+'/te_sample.csv', header=0)
    valist = pd.read_csv(totlist_dir+'/va_sample.csv', header=0)
    trimlista = trlist['L0path'].values.tolist()
    trimlistb = trlist['L1path'].values.tolist()
    trimlistc = trlist['L2path'].values.tolist()
    trlblist = trlist['label'].values.tolist()
    teimlista = telist['L0path'].values.tolist()
    teimlistb = telist['L1path'].values.tolist()
    teimlistc = telist['L2path'].values.tolist()
    telblist = telist['label'].values.tolist()
    vaimlista = valist['L0path'].values.tolist()
    vaimlistb = valist['L1path'].values.tolist()
    vaimlistc = valist['L2path'].values.tolist()
    valblist = valist['label'].values.tolist()

    train_filename = data_dir+'/train.tfrecords'
    writer = tf.python_io.TFRecordWriter(train_filename)
    for i in range(len(trlblist)):
        if not i % 1000:
            sys.stdout.flush()
        try:
            # Load the image
            imga = load_image(trimlista[i])
            imgb = load_image(trimlistb[i])
            imgc = load_image(trimlistc[i])
            label = trlblist[i]
            # Create a feature
            feature = {'train/label': _int64_feature(label),
                       'train/imageL0': _bytes_feature(tf.compat.as_bytes(imga.tostring())),
                       'train/imageL1': _bytes_feature(tf.compat.as_bytes(imgb.tostring())),
                       'train/imageL2': _bytes_feature(tf.compat.as_bytes(imgc.tostring()))}
            # Create an example protocol buffer
            example = tf.train.Example(features=tf.train.Features(feature=feature))

            # Serialize to string and write on the file
            writer.write(example.SerializeToString())
        except AttributeError:
            print('Error image: '+ trimlista[i] + '~' + trimlistb[i] + '~' + trimlistc[i])
            pass

    writer.close()
    sys.stdout.flush()

    test_filename = data_dir+'/test.tfrecords'
    writer = tf.python_io.TFRecordWriter(test_filename)
    for i in range(len(telblist)):
        if not i % 1000:
            sys.stdout.flush()
        try:
            # Load the image
            imga = load_image(teimlista[i])
            imgb = load_image(teimlistb[i])
            imgc = load_image(teimlistc[i])
            label = telblist[i]
            # Create a feature
            feature = {'test/label': _int64_feature(label),
                       'test/imageL0': _bytes_feature(tf.compat.as_bytes(imga.tostring())),
                       'test/imageL1': _bytes_feature(tf.compat.as_bytes(imgb.tostring())),
                       'test/imageL2': _bytes_feature(tf.compat.as_bytes(imgc.tostring()))}
            # Create an example protocol buffer
            example = tf.train.Example(features=tf.train.Features(feature=feature))

            # Serialize to string and write on the file
            writer.write(example.SerializeToString())
        except AttributeError:
            print('Error image:'+ teimlista[i] + '~' + teimlistb[i] + '~' + teimlistc[i])
            pass
    writer.close()
    sys.stdout.flush()

    val_filename = data_dir+'/validation.tfrecords'
    writer = tf.python_io.TFRecordWriter(val_filename)
    for i in range(len(valblist)):
        if not i % 1000:
            sys.stdout.flush()
        try:
            # Load the image
            imga = load_image(vaimlista[i])
            imgb = load_image(vaimlistb[i])
            imgc = load_image(vaimlistc[i])
            label = valblist[i]
            # Create a feature
            feature = {'validation/label': _int64_feature(label),
                       'validation/imageL0': _bytes_feature(tf.compat.as_bytes(imga.tostring())),
                       'validation/imageL1': _bytes_feature(tf.compat.as_bytes(imgb.tostring())),
                       'validation/imageL2': _bytes_feature(tf.compat.as_bytes(imgc.tostring()))}
            # Create an example protocol buffer
            example = tf.train.Example(features=tf.train.Features(feature=feature))

            # Serialize to string and write on the file
            writer.write(example.SerializeToString())
        except AttributeError:
            print('Error image:'+ vaimlista[i] + '~' + vaimlistb[i] + '~' + vaimlistc[i])
            pass
    writer.close()
    sys.stdout.flush()


# load tfrecords and prepare datasets
def tfreloader(mode, ep, bs, cls, ctr, cte, cva):
    filename = data_dir + '/' + mode + '.tfrecords'
    if mode == 'train':
        ct = ctr
    elif mode == 'test':
        ct = cte
    else:
        ct = cva

    datasets = data_input3.DataSet(bs, ct, ep=ep, cls=cls, mode=mode, filename=filename)

    return datasets


# main; trc is training image count; tec is testing image count; to_reload is the model to load; test or not
def main(trc, tec, vac, cls, testset=None, to_reload=None, test=None):

    if test:  # restore for testing only
        m = cnn5.INCEPTION(INPUT_DIM, HYPERPARAMS, meta_graph=to_reload, log_dir=LOG_DIR, meta_dir=LOG_DIR, model=md)
        print("Loaded! Ready for test!", flush=True)
        if tec >= bs:
            THE = tfreloader('test', 1, bs, cls, trc, tec, vac)
            m.inference(THE, dirr, testset=testset, pmd=pdmd)
        else:
            print("Not enough testing images!")

    elif to_reload:  # restore for further training and testing
        m = cnn5.INCEPTION(INPUT_DIM, HYPERPARAMS, meta_graph=to_reload, log_dir=LOG_DIR, meta_dir=LOG_DIR, model=md)
        print("Loaded! Restart training.", flush=True)
        HE = tfreloader('train', ep, bs, cls, trc, tec, vac)
        VHE = tfreloader('validation', ep*10, bs, cls, trc, tec, vac)
        itt = int(trc * ep / bs)
        if trc <= 2 * bs or vac <= bs:
            print("Not enough training/validation images!")
        else:
            m.train(HE, VHE, trc, bs, pmd=pdmd, dirr=dirr, max_iter=itt, verbose=True, save=True, outdir=METAGRAPH_DIR)
        if tec >= bs:
            THE = tfreloader('test', 1, bs, cls, trc, tec, vac)
            m.inference(THE, dirr, testset=testset, pmd=pdmd)
        else:
            print("Not enough testing images!")

    else:  # train and test
        m = cnn5.INCEPTION(INPUT_DIM, HYPERPARAMS, log_dir=LOG_DIR, model=md)
        print("Start a new training!")
        HE = tfreloader('train', ep, bs, cls, trc, tec, vac)
        VHE = tfreloader('validation', ep*10, bs, cls, trc, tec, vac)
        itt = int(trc*ep/bs)+1
        if trc <= 2 * bs or vac <= bs:
            print("Not enough training/validation images!")
        else:
            m.train(HE, VHE, trc, bs, pmd=pdmd, dirr=dirr, max_iter=itt, verbose=True, save=True, outdir=METAGRAPH_DIR)
        if tec >= bs:
            THE = tfreloader('test', 1, bs, cls, trc, tec, vac)
            m.inference(THE, dirr, testset=testset, pmd=pdmd)
        else:
            print("Not enough testing images!")


if __name__ == "__main__":
    tf.reset_default_graph()
    # make directories if not exist
    for DIR in (LOG_DIR, METAGRAPH_DIR, data_dir, out_dir):
        try:
            os.mkdir(DIR)
        except FileExistsError:
            pass
    # get counts of testing, validation, and training datasets;
    # if not exist, prepare testing and training datasets from sampling
    try:
        trc, tec, vac = counters(data_dir)
        tes = pd.read_csv(data_dir+'/te_sample.csv', header=0)
        vas = pd.read_csv(data_dir+'/va_sample.csv', header=0)
    except FileNotFoundError:
        alll = Sample_prep2.big_image_sum(pmd=pdmd, path=img_dir)
        trs, tes, vas = Sample_prep2.set_sep(alll, path=data_dir, cls=classes)
        trc, tec, vac = counters(data_dir)
        loader(data_dir)
    # have trained model or not; train from scratch if not
    try:
        modeltoload = sys.argv[6]
        # test or not
        try:
            testmode = sys.argv[7]
            main(trc, tec, vac, classes, testset=tes, to_reload=modeltoload, test=True)
        except IndexError:
            main(trc, tec, vac, classes, testset=tes, to_reload=modeltoload)
    except IndexError:
        if not os.path.isfile(data_dir + '/test.tfrecords'):
            loader(data_dir)
        if not os.path.isfile(data_dir + '/train.tfrecords'):
            loader(data_dir)
        if not os.path.isfile(data_dir + '/validation.tfrecords'):
            loader(data_dir)
        main(trc, tec, vac, classes, testset=tes)

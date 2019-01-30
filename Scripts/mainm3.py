#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main method for CPTAC-UCEC

Created on 11/26/2018

@author: RH
"""
import matplotlib
matplotlib.use('Agg')
import os
import sys
import numpy as np
import tensorflow as tf
import data_input
import cnn3
import pandas as pd
import cv2
import Sample_prep
import time


dirr = sys.argv[1]  # output directory
bs = sys.argv[2]    # batch size
ep = sys.argv[3]    # epochs to train
md = sys.argv[4]    # structure to use
level = sys.argv[5] # level of tiles to use
bs = int(bs)
ep = int(ep)

IMG_DIM = 299
# input image dimension
INPUT_DIM = [bs, IMG_DIM, IMG_DIM, 3]
# hyper parameters
HYPERPARAMS = {
    "batch_size": bs,
    "dropout": 0.8,
    "learning_rate": 1E-4
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
    trcc = len(trlist['label']) - 1
    tecc = len(telist['label']) - 1

    return trcc, tecc


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
    trimlist = trlist['path'].values.tolist()
    trlblist = trlist['label'].values.tolist()
    teimlist = telist['path'].values.tolist()
    telblist = telist['label'].values.tolist()

    train_filename = data_dir+'/train.tfrecords'
    writer = tf.python_io.TFRecordWriter(train_filename)
    for i in range(len(trimlist)):
        if not i % 1000:
            sys.stdout.flush()
        try:
            # Load the image
            img = load_image(trimlist[i])
            label = trlblist[i]
            # Create a feature
            feature = {'train/label': _int64_feature(label),
                       'train/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
            # Create an example protocol buffer
            example = tf.train.Example(features=tf.train.Features(feature=feature))

            # Serialize to string and write on the file
            writer.write(example.SerializeToString())
        except AttributeError:
            print('Error image:'+trimlist[i])
            pass

    writer.close()
    sys.stdout.flush()

    test_filename = data_dir+'/test.tfrecords'
    writer = tf.python_io.TFRecordWriter(test_filename)
    for i in range(len(teimlist)):
        if not i % 1000:
            sys.stdout.flush()
        try:
            # Load the image
            img = load_image(teimlist[i])
            label = telblist[i]
            # Create a feature
            feature = {'test/label': _int64_feature(label),
                       'test/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
            # Create an example protocol buffer
            example = tf.train.Example(features=tf.train.Features(feature=feature))

            # Serialize to string and write on the file
            writer.write(example.SerializeToString())
        except AttributeError:
            print('Error image:'+trimlist[i])
            pass
    writer.close()
    sys.stdout.flush()


# load tfrecords and prepare datasets
def tfreloader(mode, ep, bs, ctr, cte):
    filename = data_dir + '/' + mode + '.tfrecords'
    if mode == 'train':
        ct = ctr
    else:
        ct = cte

    datasets = data_input.DataSet(bs, ct, ep=ep, mode=mode, filename=filename)

    return datasets


# main; trc is training image count; tec is testing image count; to_reload is the model to load; test or not
def main(trc, tec, to_reload=None, test=None):

    if test:  # restore for testing only
        m = cnn3.INCEPTION(INPUT_DIM, HYPERPARAMS, meta_graph=to_reload, log_dir=LOG_DIR, meta_dir=LOG_DIR, model=md)
        print("Loaded! Ready for test!", flush=True)
        if tec >= 1000:
            HE = tfreloader('test', 1, 1000, trc, tec)
            m.inference(HE, dirr)
        elif 100 < tec < 1000:
            HE = tfreloader('test', 1, tec, trc, tec)
            m.inference(HE, dirr)
        else:
            print("Not enough testing images!")

    elif to_reload:  # restore for further training and testing
        m = cnn3.INCEPTION(INPUT_DIM, HYPERPARAMS, meta_graph=to_reload, log_dir=LOG_DIR, meta_dir=LOG_DIR, model=md)
        print("Loaded! Restart training.", flush=True)
        HE = tfreloader('train', ep, bs, trc, tec)
        itt = int(trc * ep / bs)
        if trc <= 2 * bs:
            print("Not enough training images!")
        else:
            m.train(HE, trc, bs, dirr=dirr, max_iter=itt, verbose=True, save=True, outdir=METAGRAPH_DIR)
        if tec >= 1000:
            HE = tfreloader('test', 1, 1000, trc, tec)
            m.inference(HE, dirr)
        elif 100 < tec < 1000:
            HE = tfreloader('test', 1, tec, trc, tec)
            m.inference(HE, dirr)
        else:
            print("Not enough testing images!")

    else:  # train and test
        m = cnn3.INCEPTION(INPUT_DIM, HYPERPARAMS, log_dir=LOG_DIR, model=md)
        print("Start a new training!")
        HE = tfreloader('train', ep, bs, trc, tec)
        itt = int(trc*ep/bs)+1
        if trc <= 2*bs:
            print("Not enough training images!")
        else:
            m.train(HE, trc, bs, dirr=dirr, max_iter=itt, verbose=True, save=True, outdir=METAGRAPH_DIR)
        if tec >= 1000:
            HE = tfreloader('test', 1, 1000, trc, tec)
            m.inference(HE, dirr)
        elif 100 < tec < 1000:
            HE = tfreloader('test', 1, tec, trc, tec)
            m.inference(HE, dirr)
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
    # get counts of testing and training dataset; if not exist, prepare testing and training datasets from sampling
    try:
        trc, tec = counters(data_dir)
    except FileNotFoundError:
        alll = Sample_prep.big_image_sum(level, path=img_dir)
        tes, trs = Sample_prep.set_sep(alll, level, path=data_dir)
        trc, tec = counters(data_dir)
        loader(data_dir)
    # have trained model or not; train from scratch if not
    try:
        modeltoload = sys.argv[6]
        # test or not
        try:
            testmode = sys.argv[7]
            main(trc, tec, to_reload=modeltoload, test=True)
        except IndexError:
            main(trc, tec, to_reload=modeltoload)
    except IndexError:
        if not os.path.isfile(data_dir + '/test.tfrecords'):
            loader(data_dir)
        main(trc, tec)



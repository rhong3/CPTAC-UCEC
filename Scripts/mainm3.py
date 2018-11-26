#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 11/15/2018

@author: RH
"""
import matplotlib
matplotlib.use('Agg')
import os
import sys
import numpy as np
import tensorflow as tf
import data_input
import cnnm3
import cnng3
import cnni3
import cnnt3
import cnnir13
import cnnir23
import pandas as pd
import cv2
import Sample_prep


dirr = sys.argv[1]
bs = sys.argv[2]
ep = sys.argv[3]
md = sys.argv[4]
bs = int(bs)
ep = int(ep)

IMG_DIM = 299

INPUT_DIM = [bs, IMG_DIM, IMG_DIM, 3]

HYPERPARAMS = {
    "batch_size": bs,
    "dropout": 0.8,
    "learning_rate": 1E-4
}

MAX_ITER = np.inf
MAX_EPOCHS = ep

img_dir = '../Neutrophil/All_Tiles_final'
LOG_DIR = "../Neutrophil/{}".format(dirr)
METAGRAPH_DIR = "../Neutrophil/{}".format(dirr)
data_dir = "../Neutrophil/{}/data".format(dirr)
out_dir = "../Neutrophil/{}/out".format(dirr)


def counters(totlist_dir):
    trlist = pd.read_csv(totlist_dir + '/tr_sample.csv', header=0)
    telist = pd.read_csv(totlist_dir + '/te_sample.csv', header=0)
    trcc = len(trlist['label']) - 1
    tecc = len(telist['label']) - 1

    return trcc, tecc


def load_image(addr):
    img = cv2.imread(addr)
    img = img.astype(np.float32)
    return img


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


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

    writer.close()
    sys.stdout.flush()

    test_filename = data_dir+'/test.tfrecords'
    writer = tf.python_io.TFRecordWriter(test_filename)
    for i in range(len(teimlist)):
        if not i % 1000:
            sys.stdout.flush()
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

    writer.close()
    sys.stdout.flush()


def tfreloader(mode, ep, bs, ctr, cte):
    filename = data_dir + '/' + mode + '.tfrecords'
    if mode == 'train':
        ct = ctr
    else:
        ct = cte

    datasets = data_input.DataSet(bs, ct, ep=ep, mode=mode, filename=filename)

    return datasets


def main(trc, tec, to_reload=None, test=None):

    if test:  # restore

        if md == 'IG':
            m = cnng3.INCEPTION(INPUT_DIM, HYPERPARAMS, meta_graph=to_reload, log_dir=LOG_DIR, meta_dir=LOG_DIR)
        elif md == 'I2':
            m = cnnt3.INCEPTION(INPUT_DIM, HYPERPARAMS, meta_graph=to_reload, log_dir=LOG_DIR, meta_dir=LOG_DIR)
        elif md == 'I3':
            m = cnnm3.INCEPTION(INPUT_DIM, HYPERPARAMS, meta_graph=to_reload, log_dir=LOG_DIR, meta_dir=LOG_DIR)
        elif md == 'I4':
            m = cnni3.INCEPTION(INPUT_DIM, HYPERPARAMS, meta_graph=to_reload, log_dir=LOG_DIR, meta_dir=LOG_DIR)
        elif md == 'IR1':
            m = cnnir13.INCEPTION(INPUT_DIM, HYPERPARAMS, meta_graph=to_reload, log_dir=LOG_DIR, meta_dir=LOG_DIR)
        elif md == 'IR2':
            m = cnnir23.INCEPTION(INPUT_DIM, HYPERPARAMS, meta_graph=to_reload, log_dir=LOG_DIR, meta_dir=LOG_DIR)
        else:
            m = cnng3.INCEPTION(INPUT_DIM, HYPERPARAMS, meta_graph=to_reload, log_dir=LOG_DIR, meta_dir=LOG_DIR)

        print("Loaded! Ready for test!", flush=True)
        if tec >= 1000:
            HE = tfreloader('test', 1, 1000, trc, tec)
            m.inference(HE, dirr)
        elif 100 < tec < 1000:
            HE = tfreloader('test', 1, tec, trc, tec)
            m.inference(HE, dirr)
        else:
            print("Not enough testing images!")

    elif to_reload:

        if md == 'IG':
            m = cnng3.INCEPTION(INPUT_DIM, HYPERPARAMS, meta_graph=to_reload, log_dir=LOG_DIR, meta_dir=LOG_DIR)
        elif md == 'I2':
            m = cnnt3.INCEPTION(INPUT_DIM, HYPERPARAMS, meta_graph=to_reload, log_dir=LOG_DIR, meta_dir=LOG_DIR)
        elif md == 'I3':
            m = cnnm3.INCEPTION(INPUT_DIM, HYPERPARAMS, meta_graph=to_reload, log_dir=LOG_DIR, meta_dir=LOG_DIR)
        elif md == 'I4':
            m = cnni3.INCEPTION(INPUT_DIM, HYPERPARAMS, meta_graph=to_reload, log_dir=LOG_DIR, meta_dir=LOG_DIR)
        elif md == 'IR1':
            m = cnnir13.INCEPTION(INPUT_DIM, HYPERPARAMS, meta_graph=to_reload, log_dir=LOG_DIR, meta_dir=LOG_DIR)
        elif md == 'IR2':
            m = cnnir23.INCEPTION(INPUT_DIM, HYPERPARAMS, meta_graph=to_reload, log_dir=LOG_DIR, meta_dir=LOG_DIR)
        else:
            m = cnng3.INCEPTION(INPUT_DIM, HYPERPARAMS, meta_graph=to_reload, log_dir=LOG_DIR, meta_dir=LOG_DIR)

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

    else:  # train

        if md == 'IG':
            m = cnng3.INCEPTION(INPUT_DIM, HYPERPARAMS, log_dir=LOG_DIR)
        elif md == 'I2':
            m = cnnt3.INCEPTION(INPUT_DIM, HYPERPARAMS, log_dir=LOG_DIR)
        elif md == 'I3':
            m = cnnm3.INCEPTION(INPUT_DIM, HYPERPARAMS, log_dir=LOG_DIR)
        elif md == 'I4':
            m = cnni3.INCEPTION(INPUT_DIM, HYPERPARAMS, log_dir=LOG_DIR)
        elif md == 'IR1':
            m = cnnir13.INCEPTION(INPUT_DIM, HYPERPARAMS, log_dir=LOG_DIR)
        elif md == 'IR2':
            m = cnnir23.INCEPTION(INPUT_DIM, HYPERPARAMS, log_dir=LOG_DIR)
        else:
            m = cnng3.INCEPTION(INPUT_DIM, HYPERPARAMS, log_dir=LOG_DIR)

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

    for DIR in (LOG_DIR, METAGRAPH_DIR, data_dir, out_dir):
        try:
            os.mkdir(DIR)
        except(FileExistsError):
            pass

    _, _, _, tes, trs = Sample_prep.samplesum()
    tes.to_csv(img_dir+'/te_sample.csv', index=False)
    trs.to_csv(img_dir+'/tr_sample.csv', index=False)
    trc, tec = counters(img_dir)

    try:
        modeltoload = sys.argv[5]
        try:
            testmode = sys.argv[6]
            main(trc, tec, to_reload=modeltoload, test=True)
        except(IndexError):
            main(trc, tec, to_reload=modeltoload)
    except(IndexError):
        if not os.path.isfile(data_dir + '/test.tfrecords'):
            loader(img_dir)
        main(trc, tec)



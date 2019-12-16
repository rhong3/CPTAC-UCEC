"""
load a trained model and apply it

Created on 12/16/2019

@author: RH
"""
import Slicer
import time
import matplotlib
matplotlib.use('Agg')
import os
import sys
import numpy as np
import pandas as pd
import cv2
import skimage.morphology as mph
import tensorflow as tf


start_time = time.time()

dirr = sys.argv[1]  # name of output directory
imgfile = sys.argv[2]  # input scn/svs name
bs = sys.argv[3]  # batch size
bs = int(bs)
md = sys.argv[4]  # loaded model's architecture
modeltoload = sys.argv[5]  # full path to the trained model to be loaded
pdmd = sys.argv[6]  # feature to predict


# read images
def load_image(addr):
    img = cv2.imread(addr)
    img = img.astype(np.float32)
    return img


# used for tfrecord float generation
def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


# used for tfrecord labels generation
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# used for tfrecord images generation
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# loading images for dictionaries and generate tfrecords
def loaderX(totlist_dir, ds):
    if ds == 'train':
        slist = pd.read_csv(totlist_dir + '/tr_sample.csv', header=0)
    elif ds == 'validation':
        slist = pd.read_csv(totlist_dir + '/va_sample.csv', header=0)
    elif ds == 'test':
        slist = pd.read_csv(totlist_dir + '/te_sample.csv', header=0)
    else:
        slist = pd.read_csv(totlist_dir + '/te_sample.csv', header=0)
    imlista = slist['L0path'].values.tolist()
    imlistb = slist['L1path'].values.tolist()
    imlistc = slist['L2path'].values.tolist()
    lblist = slist['label'].values.tolist()
    wtlist = slist['BMI'].values.tolist()
    aglist = slist['age'].values.tolist()
    filename = data_dir + '/' + ds + '.tfrecords'
    writer = tf.python_io.TFRecordWriter(filename)
    for i in range(len(lblist)):
        try:
            # Load the image
            imga = load_image(imlista[i])
            imgb = load_image(imlistb[i])
            imgc = load_image(imlistc[i])
            label = lblist[i]
            wt = wtlist[i]
            ag = aglist[i]
            # Create a feature
            feature = {ds + '/label': _int64_feature(label),
                       ds + '/BMI': _float_feature(wt),
                       ds + '/age': _float_feature(ag),
                       ds + '/imageL0': _bytes_feature(tf.compat.as_bytes(imga.tostring())),
                       ds + '/imageL1': _bytes_feature(tf.compat.as_bytes(imgb.tostring())),
                       ds + '/imageL2': _bytes_feature(tf.compat.as_bytes(imgc.tostring()))}
            # Create an example protocol buffer
            example = tf.train.Example(features=tf.train.Features(feature=feature))

            # Serialize to string and write on the file
            writer.write(example.SerializeToString())
        except AttributeError:
            print('Error image: ' + imlista[i] + '~' + imlistb[i] + '~' + imlistc[i])
            pass

    writer.close()


# loading images for dictionaries and generate tfrecords
def loaderI(totlist_dir, ds):
    if ds == 'train':
        slist = pd.read_csv(totlist_dir + '/tr_sample.csv', header=0)
    elif ds == 'validation':
        slist = pd.read_csv(totlist_dir + '/va_sample.csv', header=0)
    elif ds == 'test':
        slist = pd.read_csv(totlist_dir + '/te_sample.csv', header=0)
    else:
        slist = pd.read_csv(totlist_dir + '/te_sample.csv', header=0)
    imlist = slist['path'].values.tolist()
    lblist = slist['label'].values.tolist()

    filename = data_dir + '/' + ds + '.tfrecords'
    writer = tf.python_io.TFRecordWriter(filename)
    for i in range(len(imlist)):
        if not i % 1000:
            sys.stdout.flush()
        try:
            # Load the image
            img = load_image(imlist[i])
            label = lblist[i]
            # Create a feature
            feature = {ds + '/label': _int64_feature(label),
                       ds + '/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
            # Create an example protocol buffer
            example = tf.train.Example(features=tf.train.Features(feature=feature))

            # Serialize to string and write on the file
            writer.write(example.SerializeToString())
        except AttributeError:
            print('Error image:' + imlist[i])
            pass

    writer.close()
    sys.stdout.flush()


if 'I' in md:
    import Sample_prep as Sample_prep
    import cnn4 as cnn
    import data_input2 as data_input
    level = 1
    loader = loaderI
else:
    import Sample_prep2 as Sample_prep
    import cnn5 as cnn
    import data_input_fusion as data_input
    level = None
    loader = loaderX

try:
    sup = sys.argv[7]  # fusion mode
except IndexError:
    sup = False

if pdmd == 'subtype':
    classes = 4
else:
    classes = 2

print('Input config:')
print(dirr, imgfile, bs, md, pdmd, sup)


# input image dimension
INPUT_DIM = [bs, 299, 299, 3]
# hyper parameters
HYPERPARAMS = {
    "batch_size": bs,
    "dropout": 0.3,
    "learning_rate": 1E-4,
    "classes": classes,
    "sup": sup
}

# paths to directories
img_dir = '../tiles/'
LOG_DIR = "../Results/{}".format(dirr)
METAGRAPH_DIR = "../Results/{}".format(dirr)
data_dir = "../Results/{}/data".format(dirr)
out_dir = "../Results/{}/out".format(dirr)


# load tfrecords and prepare datasets
def tfreloader(bs, cls, ct):
    filename = data_dir + '/test.tfrecords'
    datasets = data_input.DataSet(bs, ct, ep=1, cls=cls, mode='test', filename=filename)

    return datasets


# main function for real test image prediction
def test(bs, cls, to_reload):
    m = cnn.INCEPTION(INPUT_DIM, HYPERPARAMS, meta_graph=to_reload, log_dir=LOG_DIR, meta_dir=LOG_DIR, model=md)

    print("Loaded! Ready for test!")
    HE = tfreloader(bs, cls, None)
    m.inference(HE, dirr, Not_Realtest=False, bs=bs, pmd=pdmd)


if __name__ == "__main__":
    # make directories if not exist
    for DIR in (LOG_DIR, METAGRAPH_DIR, data_dir, out_dir):
        try:
            os.mkdir(DIR)
        except FileExistsError:
            pass
    if "TCGA" in imgfile:


"""
Main method for X
Created on 04/26/2019; modified 11/06/2019
@author: RH
"""
import os
import sys
import numpy as np
import tensorflow as tf
import cnn5
import Sample_prep2
import pandas as pd
import cv2
import data_input_fusion as data_input3
import time
import matplotlib
matplotlib.use('Agg')


# count numbers of training and testing images
def counters(totlist_dir, cls):
    trlist = pd.read_csv(totlist_dir + '/tr_sample.csv', header=0)
    telist = pd.read_csv(totlist_dir + '/te_sample.csv', header=0)
    valist = pd.read_csv(totlist_dir + '/va_sample.csv', header=0)
    trcc = len(trlist['label'])
    tecc = len(telist['label'])
    vacc = len(valist['label'])
    weigh = []
    for i in range(cls):
        ccct = len(trlist.loc[trlist['label'] == i])+len(valist.loc[valist['label'] == i])\
               + len(telist.loc[telist['label'] == i])
        wt = ((trcc+tecc+vacc)/cls)/ccct
        weigh.append(wt)
    weigh = tf.constant(weigh)
    return trcc, tecc, vacc, weigh


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
def loader(totlist_dir, ds):
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
    filename = totlist_dir + '/' + ds + '.tfrecords'
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


dirr = sys.argv[1]  # output directory
bs = sys.argv[2]  # batch size
bs = int(bs)
md = sys.argv[3]  # architecture to use
pdmd = sys.argv[4]  # feature to predict

try:
    ep = sys.argv[5]  # epochs to train
    ep = int(ep)
except IndexError:
    ep = 100

try:
    sup = sys.argv[6]  # fusion mode
    if sup == "True":
        sup = True
    else:
        sup = False
except IndexError:
    sup = False

if pdmd == 'subtype':
    classes = 4
else:
    classes = 2

print('Input config:')
print(dirr, bs, md, pdmd, ep, sup)

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


# main; trc is training image count; tec is testing image count; to_reload is the model to load; test or not
def main(trc, tec, vac, cls, weight, testset=None, to_reload=None, test=None):

    if test:  # restore for testing only
        m = cnn5.INCEPTION(INPUT_DIM, HYPERPARAMS, meta_graph=to_reload, log_dir=LOG_DIR,
                           meta_dir=LOG_DIR, model=md, weights=weight)
        print("Loaded! Ready for test!")
        if tec >= bs:
            THE = tfreloader('test', 1, bs, cls, trc, tec, vac)
            m.inference(THE, dirr, testset=testset, pmd=pdmd)
        else:
            print("Not enough testing images!")

    elif to_reload:  # restore for further training and testing
        m = cnn5.INCEPTION(INPUT_DIM, HYPERPARAMS, meta_graph=to_reload, log_dir=LOG_DIR,
                           meta_dir=LOG_DIR, model=md, weights=weight)
        print("Loaded! Restart training.")
        HE = tfreloader('train', ep, bs, cls, trc, tec, vac)
        VHE = tfreloader('validation', ep*100, bs, cls, trc, tec, vac)
        itt = int(trc * ep / bs)
        if trc <= 2 * bs or vac <= bs:
            print("Not enough training/validation images!")
        else:
            m.train(HE, VHE, trc, bs, pmd=pdmd, dirr=dirr, max_iter=itt, save=True, outdir=METAGRAPH_DIR)
        if tec >= bs:
            THE = tfreloader('test', 1, bs, cls, trc, tec, vac)
            m.inference(THE, dirr, testset=testset, pmd=pdmd)
        else:
            print("Not enough testing images!")

    else:  # train and test
        m = cnn5.INCEPTION(INPUT_DIM, HYPERPARAMS, log_dir=LOG_DIR, model=md, weights=weight)
        print("Start a new training!")
        HE = tfreloader('train', ep, bs, cls, trc, tec, vac)
        VHE = tfreloader('validation', ep*100, bs, cls, trc, tec, vac)
        itt = int(trc*ep/bs)+1
        if trc <= 2 * bs or vac <= bs:
            print("Not enough training/validation images!")
        else:
            m.train(HE, VHE, trc, bs, pmd=pdmd, dirr=dirr, max_iter=itt, save=True, outdir=METAGRAPH_DIR)
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
        trc, tec, vac, weights = counters(data_dir, classes)
        trs = pd.read_csv(data_dir + '/tr_sample.csv', header=0)
        tes = pd.read_csv(data_dir+'/te_sample.csv', header=0)
        vas = pd.read_csv(data_dir+'/va_sample.csv', header=0)
    except FileNotFoundError:
        alll = Sample_prep2.big_image_sum(pmd=pdmd, path=img_dir)
        # trs, tes, vas = Sample_prep2.set_sep_secondary(alll, path=data_dir, cls=classes, pmd=pdmd, batchsize=bs)
        trs, tes, vas = Sample_prep2.set_sep_idp(alll, path=data_dir, cls=classes, batchsize=bs)
        # trs, tes, vas = Sample_prep2.set_sep(alll, path=data_dir, cls=classes, batchsize=bs)
        trc, tec, vac, weights = counters(data_dir, classes)
        loader(data_dir, 'train')
        loader(data_dir, 'validation')
        loader(data_dir, 'test')
    # have trained model or not; train from scratch if not
    try:
        modeltoload = sys.argv[7]
        # test or not
        try:
            testmode = sys.argv[8]
            main(trc, tec, vac, classes, weights, testset=tes, to_reload=modeltoload, test=True)
        except IndexError:
            main(trc, tec, vac, classes, weights, testset=tes, to_reload=modeltoload)
    except IndexError:
        if not os.path.isfile(data_dir + '/test.tfrecords'):
            loader(data_dir, 'test')
        if not os.path.isfile(data_dir + '/train.tfrecords'):
            loader(data_dir, 'train')
        if not os.path.isfile(data_dir + '/validation.tfrecords'):
            loader(data_dir, 'validation')
        if sup:
            print("Using Fusion Mode!")
        main(trc, tec, vac, classes, weights, testset=tes)

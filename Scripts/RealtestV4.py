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
import sklearn.utils as sku
import tensorflow as tf
import staintools
import re


start_time = time.time()

dirr = sys.argv[1]  # name of output directory
imgfile = sys.argv[2]  # input scn/svs name (eg. TCGA/XXX.scn)
bs = sys.argv[3]  # batch size
bs = int(bs)
md = sys.argv[4]  # loaded model's architecture
modeltoload = sys.argv[5]  # full path to the trained model to be loaded
pdmd = sys.argv[6]  # feature to predict

if 'I' in md:
    import cnn4 as cnn
    import data_input2 as data_input
    level = 1
    loader = loaderI
    cut = 2
else:
    import cnn5 as cnn
    import data_input3 as data_input
    level = None
    loader = loaderX
    cut = 4

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
LOG_DIR = "../Results/{}".format(dirr)
METAGRAPH_DIR = "../Results/{}".format(dirr)
data_dir = "../Results/{}".format(dirr)


# pair tiles of 10x, 5x, 2.5x of the same area
def paired_tile_ids_in(root_dir):
    dira = os.path.isdir(root_dir + 'level1')
    dirb = os.path.isdir(root_dir + 'level2')
    dirc = os.path.isdir(root_dir + 'level3')
    if dira and dirb and dirc:
        if "TCGA" in root_dir:
            fac = 2000
        else:
            fac = 1000
        ids = []
        for level in range(1, 4):
            dirr = root_dir + 'level{}'.format(str(level))
            for id in os.listdir(dirr):
                if '.png' in id:
                    x = int(float(id.split('x-', 1)[1].split('-', 1)[0]) / fac)
                    y = int(float(re.split('_', id.split('y-', 1)[1])[0]) / fac)
                    try:
                        dup = re.split('.p', re.split('_', id.split('y-', 1)[1])[1])[0]
                    except IndexError:
                        dup = np.nan
                    ids.append([level, dirr + '/' + id, x, y, dup])
                else:
                    print('Skipping ID:', id)
        ids = pd.DataFrame(ids, columns=['level', 'path', 'x', 'y', 'dup'])
        idsa = ids.loc[ids['level'] == 1]
        idsa = idsa.drop(columns=['level'])
        idsa = idsa.rename(index=str, columns={"path": "L0path"})
        idsb = ids.loc[ids['level'] == 2]
        idsb = idsb.drop(columns=['level'])
        idsb = idsb.rename(index=str, columns={"path": "L1path"})
        idsc = ids.loc[ids['level'] == 3]
        idsc = idsc.drop(columns=['level'])
        idsc = idsc.rename(index=str, columns={"path": "L2path"})
        idsa = pd.merge(idsa, idsb, on=['x', 'y', 'dup'], how='left', validate="many_to_many")
        idsa['x'] = idsa['x'] - (idsa['x'] % 2)
        idsa['y'] = idsa['y'] - (idsa['y'] % 2)
        idsa = pd.merge(idsa, idsc, on=['x', 'y', 'dup'], how='left', validate="many_to_many")
        idsa = idsa.drop(columns=['x', 'y', 'dup'])
        idsa = idsa.dropna()
        idsa = sku.shuffle(idsa)
    else:
        idsa = pd.DataFrame(columns=['L0path', 'L1path', 'L2path'])

    return idsa


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
def loaderX(totlist_dir):
    slist = pd.read_csv(totlist_dir + '/te_sample.csv', header=0)
    imlista = slist['L0path'].values.tolist()
    imlistb = slist['L1path'].values.tolist()
    imlistc = slist['L2path'].values.tolist()
    filename = data_dir + '/test.tfrecords'
    writer = tf.python_io.TFRecordWriter(filename)
    for i in range(len(imlista)):
        try:
            # Load the image
            imga = load_image(imlista[i])
            imgb = load_image(imlistb[i])
            imgc = load_image(imlistc[i])
            # Create a feature
            feature = {'test/imageL0': _bytes_feature(tf.compat.as_bytes(imga.tostring())),
                       'test/imageL1': _bytes_feature(tf.compat.as_bytes(imgb.tostring())),
                       'test/imageL2': _bytes_feature(tf.compat.as_bytes(imgc.tostring()))}
            # Create an example protocol buffer
            example = tf.train.Example(features=tf.train.Features(feature=feature))

            # Serialize to string and write on the file
            writer.write(example.SerializeToString())
        except AttributeError:
            print('Error image: ' + imlista[i] + '~' + imlistb[i] + '~' + imlistc[i])
            pass

    writer.close()


# loading images for dictionaries and generate tfrecords
def loaderI(totlist_dir):
    slist = pd.read_csv(totlist_dir + '/te_sample.csv', header=0)
    imlist = slist['path'].values.tolist()
    filename = data_dir + '/test.tfrecords'
    writer = tf.python_io.TFRecordWriter(filename)
    for i in range(len(imlist)):
        try:
            # Load the image
            img = load_image(imlist[i])
            # Create a feature
            feature = {'test/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
            # Create an example protocol buffer
            example = tf.train.Example(features=tf.train.Features(feature=feature))

            # Serialize to string and write on the file
            writer.write(example.SerializeToString())
        except AttributeError:
            print('Error image:' + imlist[i])
            pass

    writer.close()


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
    for DIR in (LOG_DIR, METAGRAPH_DIR, data_dir):
        try:
            os.mkdir(DIR)
        except FileExistsError:
            pass
    # load standard image for normalization
    std = staintools.read_image("../colorstandard.png")
    std = staintools.LuminosityStandardizer.standardize(std)
    if "TCGA" in imgfile:
        for m in range(1, cut):
            level = int(m/3 + 1)
            tff = int(m/level)
            otdir = "../Results/{}/level{}".format(dirr, str(m))
            try:
                os.mkdir(otdir)
            except(FileExistsError):
                pass
            try:
                n_x, n_y, raw_img, resx, resy, ct = Slicer.tile(image_file=imgfile, outdir=otdir,
                                                                level=level, std_img=std, ft=tff)
            except Exception as e:
                print('Error!')
                pass
    else:
        for m in range(1, cut):
            level = int(m/2)
            tff = int(m % 2 + 1)
            otdir = "../Results/{}/level{}".format(dirr, str(m))
            try:
                os.mkdir(otdir)
            except(FileExistsError):
                pass
            try:
                n_x, n_y, raw_img, resx, resy, ct = Slicer.tile(image_file=imgfile, outdir=otdir,
                                                                level=level, std_img=std, ft=tff)
                if m == 1:
                    numx = n_x
                    numy = n_y
                    raw = raw_img
                    residualx = resx
                    residualy = resy
                    tct = ct
            except(IndexError):
                pass
    if not os.path.isfile(data_dir + '/test.tfrecords'):
        loader(data_dir)

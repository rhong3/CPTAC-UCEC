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
modeltoload = sys.argv[5]  # trained model to be loaded
meta = sys.argv[6]  # full path to the trained model to be loaded
pdmd = sys.argv[7]  # feature to predict


def tile_ids_in(root_dir, level=1):
    ids = []
    try:
        for id in os.listdir(root_dir):
            if '.png' in id:
                ids.append([level, root_dir+'/'+id])
            else:
                print('Skipping ID:', id)
    except FileNotFoundError:
        print('Ignore:', root_dir)
    test_tiles = pd.DataFrame(ids, columns=['level', 'L0path'])
    test_tiles.insert(loc=0, column='Num', value=test_tiles.index)
    return test_tiles


# pair tiles of 10x, 5x, 2.5x of the same area
def paired_tile_ids_in(root_dir):
    if "TCGA" in root_dir:
        fac = 2000
    else:
        fac = 1000
    ids = []
    for level in range(1, 4):
        dirrr = root_dir + '/level{}'.format(str(level))
        for id in os.listdir(dirrr):
            if '.png' in id:
                x = int(float(id.split('x-', 1)[1].split('-', 1)[0]) / fac)
                y = int(float(re.split('.p', id.split('y-', 1)[1])[0]) / fac)
                ids.append([level, dirrr + '/' + id, x, y])
            else:
                print('Skipping ID:', id)
    ids = pd.DataFrame(ids, columns=['level', 'path', 'x', 'y'])
    idsa = ids.loc[ids['level'] == 1]
    idsa = idsa.drop(columns=['level'])
    idsa = idsa.rename(index=str, columns={"path": "L0path"})
    idsb = ids.loc[ids['level'] == 2]
    idsb = idsb.drop(columns=['level'])
    idsb = idsb.rename(index=str, columns={"path": "L1path"})
    idsc = ids.loc[ids['level'] == 3]
    idsc = idsc.drop(columns=['level'])
    idsc = idsc.rename(index=str, columns={"path": "L2path"})
    idsa = pd.merge(idsa, idsb, on=['x', 'y'], how='left', validate="many_to_many")
    idsa['x'] = idsa['x'] - (idsa['x'] % 2)
    idsa['y'] = idsa['y'] - (idsa['y'] % 2)
    idsa = pd.merge(idsa, idsc, on=['x', 'y'], how='left', validate="many_to_many")
    idsa = idsa.drop(columns=['x', 'y'])
    idsa = idsa.dropna()

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
    slist = paired_tile_ids_in(totlist_dir)
    slist.insert(loc=0, column='Num', value=slist.index)
    slist.to_csv(totlist_dir + '/te_sample.csv', header=True, index=False)
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
    slist = tile_ids_in(totlist_dir)
    slist.to_csv(totlist_dir + '/te_sample.csv', header=True, index=False)
    imlist = slist['L0path'].values.tolist()
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
print(dirr, imgfile, bs, md, pdmd, sup, modeltoload, meta)


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
METAGRAPH_DIR = "../Results/{}".format(meta)
data_dir = "../Results/{}".format(dirr)
out_dir = "../Results/{}/out".format(dirr)


# load tfrecords and prepare datasets
def tfreloader(bs, cls, ct):
    filename = data_dir + '/test.tfrecords'
    datasets = data_input.DataSet(bs, ct, ep=1, cls=cls, mode='test', filename=filename)

    return datasets


# main function for real test image prediction
def test(bs, cls, to_reload):
    m = cnn.INCEPTION(INPUT_DIM, HYPERPARAMS, meta_graph=to_reload, log_dir=LOG_DIR, meta_dir=METAGRAPH_DIR, model=md)

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
                numx, numy, raw, residualx, residualy, tct = Slicer.tile(image_file=imgfile, outdir=otdir,
                                                                         level=level, std_img=std, ft=tff)
                if m == 1:
                    n_x = numx
                    n_y = numy
                    raw_img = raw
                    resx = residualx
                    resy = residualy
                    ct = tct
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
                numx, numy, raw, residualx, residualy, tct = Slicer.tile(image_file=imgfile, outdir=otdir,
                                                                level=level, std_img=std, ft=tff)
                if m == 1:
                    n_x = numx
                    n_y = numy
                    raw_img = raw
                    resx = residualx
                    resy = residualy
                    ct = tct
            except Exception as e:
                print('Error!')
                pass
    if not os.path.isfile(data_dir + '/test.tfrecords'):
        loader(data_dir)
    test(bs, classes, to_reload=modeltoload)
    slist = pd.read_csv(data_dir + '/te_sample.csv', header=0)
    # load dictionary of predictions on tiles
    teresult = pd.read_csv(out_dir+'/Test.csv', header=0)
    # join 2 dictionaries
    joined = pd.merge(slist, teresult, how='inner', on=['Num'])
    joined = joined.drop(columns=['Num'])
    tile_dict = pd.read_csv(data_dir+'/level1/dict.csv', header=0)
    tile_dict = tile_dict.rename(index=str, columns={"Loc": "L0path"})
    joined_dict = pd.merge(joined, tile_dict, how='inner', on=['L0path'])

    # save joined dictionary
    joined_dict.to_csv(out_dir + '/finaldict.csv', index=False)

    # output heat map of pos and neg.
    # initialize a graph and for each RGB channel
    opt = np.full((n_x, n_y), 0)
    hm_R = np.full((n_x, n_y), 0)
    hm_G = np.full((n_x, n_y), 0)
    hm_B = np.full((n_x, n_y), 0)

    print(np.shape(opt))

    lbdict = {0: 'negative', 1: pdmd}
    # Positive is labeled red in output heat map
    for index, row in joined_dict.iterrows():
        opt[int(row["X_pos"]), int(row["Y_pos"])] = 255
        hm_R[int(row["X_pos"]), int(row["Y_pos"])] = 255
        hm_G[int(row["X_pos"]), int(row["Y_pos"])] = int((1 - (row["POS_score"])) * 255)
        hm_B[int(row["X_pos"]), int(row["Y_pos"])] = int((1 - (row["POS_score"])) * 255)
    # expand 5 times
    opt = opt.repeat(5, axis=0).repeat(5, axis=1)
    # remove small pieces
    opt = mph.remove_small_objects(opt.astype(bool), min_size=500, connectivity=2).astype(np.uint8)

    # small-scaled original image
    ori_img = cv2.resize(raw_img, (np.shape(opt)[0] + resx, np.shape(opt)[1] + resy))
    ori_img = ori_img[:np.shape(opt)[1], :np.shape(opt)[0], :3]
    tq = ori_img[:, :, 0]
    ori_img[:, :, 0] = ori_img[:, :, 2]
    ori_img[:, :, 2] = tq
    cv2.imwrite(out_dir + '/Original_scaled.png', ori_img)

    # binary output image
    topt = np.transpose(opt)
    opt = np.full((np.shape(topt)[0], np.shape(topt)[1], 3), 0)
    opt[:, :, 0] = topt
    opt[:, :, 1] = topt
    opt[:, :, 2] = topt
    cv2.imwrite(out_dir + '/Mask.png', opt * 255)

    # output heatmap
    hm_R = np.transpose(hm_R)
    hm_G = np.transpose(hm_G)
    hm_B = np.transpose(hm_B)
    hm_R = hm_R.repeat(5, axis=0).repeat(5, axis=1)
    hm_G = hm_G.repeat(5, axis=0).repeat(5, axis=1)
    hm_B = hm_B.repeat(5, axis=0).repeat(5, axis=1)
    hm = np.dstack([hm_B, hm_G, hm_R])
    hm = hm * opt
    cv2.imwrite(out_dir + '/HM.png', hm)

    # superimpose heatmap on scaled original image
    ori_img = ori_img * opt
    overlay = ori_img * 0.65 + hm * 0.35
    cv2.imwrite(out_dir + '/Overlay.png', overlay)

    # # Time measure tool
    # start_time = time.time()
    # print("--- %s seconds ---" % (time.time() - start_time))
    print("--- %s seconds ---" % (time.time() - start_time))


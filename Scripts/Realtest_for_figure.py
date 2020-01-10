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
import tensorflow as tf
import staintools
import re
from openslide import OpenSlide

def tile_ids_in(root_dir, level=1):
    ids = []
    try:
        for id in os.listdir(root_dir+'/level1'):
            if '.png' in id:
                ids.append([level, root_dir+'/level1/'+id])
            else:
                print('Skipping ID:', id)
    except FileNotFoundError:
        print('Ignore:', root_dir)
    test_tiles = pd.DataFrame(ids, columns=['level', 'L0path'])
    test_tiles = test_tiles.reset_index(drop=True)
    test_tiles.insert(loc=0, column='Num', value=test_tiles.index)
    return test_tiles


# pair tiles of 10x, 5x, 2.5x of the same area
def paired_tile_ids_in(root_dir, imgdirr):
    if "TCGA" in imgdirr:
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
    idsa = idsa.reset_index(drop=True)

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
def loaderX(totlist_dir, imgg):
    slist = paired_tile_ids_in(totlist_dir, imgg)
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
def loaderI(totlist_dir, imgg):
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


import cnn5 as cnn
import data_input3 as data_input
level = None
loader = loaderX
cut = 4


# load tfrecords and prepare datasets
def tfreloader(bs, cls, ct):
    filename = data_dir + '/test.tfrecords'
    datasets = data_input.DataSet(bs, ct, ep=1, cls=cls, mode='test', filename=filename)

    return datasets


# main function for real test image prediction
def test(bs, cls, to_reload, LOG_DIR, METAGRAPH_DIR):
    # input image dimension
    INPUT_DIM = [bs, 299, 299, 3]
    # hyper parameters
    HYPERPARAMS = {
        "batch_size": bs,
        "dropout": 0.3,
        "learning_rate": 1E-4,
        "classes": 2,
        "sup": False
    }
    m = cnn.INCEPTION(INPUT_DIM, HYPERPARAMS, meta_graph=to_reload, log_dir=LOG_DIR, meta_dir=METAGRAPH_DIR, model=md)

    print("Loaded! Ready for test!")
    HE = tfreloader(bs, cls, None)
    m.inference(HE, meta_cutter, Not_Realtest=False, bs=bs, pmd=pdmd)


def cutter(img, outdirr, cutt):
    # load standard image for normalization
    std = staintools.read_image("../colorstandard.png")
    std = staintools.LuminosityStandardizer.standardize(std)
    if "TCGA" in img:
        for m in range(1, cutt):
            level = int(m / 3 + 1)
            tff = int(m / level)
            otdir = "../Results/{}/level{}".format(outdirr, str(m))
            try:
                os.mkdir(otdir)
            except(FileExistsError):
                pass
            try:
                numx, numy, raw, tct = Slicer.tile(image_file=img, outdir=otdir,
                                                                         level=level, std_img=std, ft=tff)
            except Exception as e:
                print('Error!')
                pass
    else:
        for m in range(1, cutt):
            level = int(m / 2)
            tff = int(m % 2 + 1)
            otdir = "../Results/{}/level{}".format(outdirr, str(m))
            try:
                os.mkdir(otdir)
            except(FileExistsError):
                pass
            try:
                numx, numy, raw, tct = Slicer.tile(image_file=imgfile, outdir=otdir,
                                                                         level=level, std_img=std, ft=tff)
            except Exception as e:
                print('Error!')
                pass


def main(dirr, imgfile, bs, md, modeltoload, meta_cut, pdmd, LOG_DIR, METAGRAPH_DIR):
    start_time = time.time()
    if pdmd == 'histology':
        pos_score = "Serous_score"
        neg_score = "Endometrioid_score"
    elif pdmd == 'MSIst':
        pos_score = "MSI-H_score"
        neg_score = "MSS_score"
    else:
        pos_score = "POS_score"
        neg_score = "NEG_score"
    # make directories if not exist
    for DIR in (LOG_DIR, METAGRAPH_DIR, data_dir, out_dir):
        try:
            os.mkdir(DIR)
        except FileExistsError:
            pass
    if "TCGA" in imgfile:
        ft = 1
        level = 1
    else:
        level = 0
        ft = 2
    slide = OpenSlide("../images/"+imgfile)

    bounds_width = slide.level_dimensions[level][0]
    bounds_height = slide.level_dimensions[level][1]
    x = 0
    y = 0
    half_width_region = 49*ft
    full_width_region = 299*ft
    stepsize = (full_width_region - half_width_region)

    n_x = int((bounds_width - 1) / stepsize)
    n_y = int((bounds_height - 1) / stepsize)

    lowres = slide.read_region((x, y), level+1, (int(n_x*stepsize/4), int(n_y*stepsize/4)))
    raw_img = np.array(lowres)[:, :, :3]
    fct = ft

    if not os.path.isfile(data_dir + '/level1/dict.csv'):
        cutter(imgfile, meta_cut, cut)

    if not os.path.isfile(data_dir + '/test.tfrecords'):
        loader(data_dir, imgfile)
    if not os.path.isfile(data_dir + '/out/Overlay.png'):
        test(bs, 2, modeltoload, LOG_DIR, METAGRAPH_DIR)
        slist = pd.read_csv(data_dir + '/te_sample.csv', header=0)
        # load dictionary of predictions on tiles
        teresult = pd.read_csv(out_dir+'/Test.csv', header=0)
        # join 2 dictionaries
        joined = pd.merge(slist, teresult, how='inner', on=['Num'])
        joined = joined.drop(columns=['Num'])
        tile_dict = pd.read_csv(data_dir+'/level1/dict.csv', header=0)
        tile_dict = tile_dict.rename(index=str, columns={"Loc": "L0path"})
        joined_dict = pd.merge(joined, tile_dict, how='inner', on=['L0path'])

        if joined_dict[pos_score].mean() > 0.5:
            print("Positive! Prediction score = " + str(joined_dict[pos_score].mean().round(5)))
        else:
            print("Negative! Prediction score = " + str(joined_dict[pos_score].mean().round(5)))
        # save joined dictionary
        joined_dict.to_csv(out_dir + '/finaldict.csv', index=False)

        # output heat map of pos and neg.
        # initialize a graph and for each RGB channel
        opt = np.full((n_x, n_y), 0)
        hm_R = np.full((n_x, n_y), 0)
        hm_G = np.full((n_x, n_y), 0)
        hm_B = np.full((n_x, n_y), 0)

        # Positive is labeled red in output heat map
        for index, row in joined_dict.iterrows():
            opt[int(row["X_pos"]), int(row["Y_pos"])] = 255
            if row[pos_score] >= 0.5:
                hm_R[int(row["X_pos"]), int(row["Y_pos"])] = 255
                hm_G[int(row["X_pos"]), int(row["Y_pos"])] = int((1-(row[pos_score]-0.5)*2)*255)
                hm_B[int(row["X_pos"]), int(row["Y_pos"])] = int((1-(row[pos_score]-0.5)*2)*255)
            else:
                hm_B[int(row["X_pos"]), int(row["Y_pos"])] = 255
                hm_G[int(row["X_pos"]), int(row["Y_pos"])] = int((1-(row[neg_score]-0.5)*2)*255)
                hm_R[int(row["X_pos"]), int(row["Y_pos"])] = int((1-(row[neg_score]-0.5)*2)*255)

        # expand 5 times
        opt = opt.repeat(50, axis=0).repeat(50, axis=1)

        # small-scaled original image
        ori_img = cv2.resize(raw_img, (np.shape(opt)[0], np.shape(opt)[1]))
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
        hm_R = hm_R.repeat(50, axis=0).repeat(50, axis=1)
        hm_G = hm_G.repeat(50, axis=0).repeat(50, axis=1)
        hm_B = hm_B.repeat(50, axis=0).repeat(50, axis=1)
        hm = np.dstack([hm_B, hm_G, hm_R])
        cv2.imwrite(out_dir + '/HM.png', hm)

        # superimpose heatmap on scaled original image
        overlay = ori_img * 0.5 + hm * 0.5
        cv2.imwrite(out_dir + '/Overlay.png', overlay)

    # # Time measure tool
    # start_time = time.time()
    # print("--- %s seconds ---" % (time.time() - start_time))
    # print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    imgdict = pd.read_csv("../Results/slide_dim.csv", header=0)

    tasks = [["his", "X1"], ["PTEN", "X1"], ["TP53", "X1"], ["MSIst", "X2"],
             ["SL", "X2"], ["ZFHX3", "X2"], ["CNVH", "X3"], ["FAT1", "X4"]]

    for aaa in tasks:
        bs = 24
        if aaa[0] == "his":
            pdmd = "histology"
        elif aaa[0] == "SL" or aaa[0] == "CNVH":
            pdmd = "Serous-like"
        else:
            pdmd = aaa[0]
        md = aaa[1]
        modeltoload = "{}_dropout_0.3".format(md)
        meta = "NL5/{}{}".format(md, aaa[0])
        result = pd.read_csv("../Results/{}/out/Test_slide.csv".format(meta), header=0)
        result = result.loc[result['True_label'] != "MSS"]
        if aaa[0] == "PTEN":
            result = result.loc[result['True_label'] == "negative"]
        else:
            result = result.loc[result['True_label'] != "negative"]
        result = result.loc[result['True_label'] != "Endometrioid"]
        result = result.loc[result['slide'] != "TCGA-B5-A0K8"]
        result = result[~result['slide'].str.contains("TCGA")]
        result = result.loc[result['True_label'] == result['Prediction']]
        todo = result["slide"].tolist()
        todotask = imgdict[imgdict['dir'].isin(todo)]
        for idx, row in todotask.iterrows():
            imgfile = "CPTAC/{}".format(row['id'])
            # paths to directories
            bigdir = "../Results/Realtest_figure"
            mediumdir = "../Results/Realtest_figure/{}".format(aaa[0])
            dirr = row['sld']
            LOG_DIR = "../Results/Realtest_figure/{}/{}".format(aaa[0], dirr)
            METAGRAPH_DIR = "../Results/{}".format(meta)
            data_dir = LOG_DIR
            out_dir = "../Results/Realtest_figure/{}/{}/out".format(aaa[0], dirr)
            for DIR in (bigdir, mediumdir, LOG_DIR, METAGRAPH_DIR, data_dir, out_dir):
                try:
                    os.mkdir(DIR)
                except FileExistsError:
                    pass
            meta_cutter = "Realtest_figure/{}/{}".format(aaa[0], dirr)
            print(meta_cutter)
            main(dirr, imgfile, bs, md, modeltoload, meta_cutter, pdmd, LOG_DIR, METAGRAPH_DIR)





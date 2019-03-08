"""
Prepare training and testing datasets as CSV dictionaries

Created on 11/26/2018

@author: RH
"""
import os
import pandas as pd
import sklearn.utils as sku
import numpy as np
import Cutter

tile_path = "../tiles/"


# get all full paths of images
def image_ids_in(root_dir, ignore=['.DS_Store','dict.csv', 'all.csv']):
    ids = []
    for id in os.listdir(root_dir):
        if id in ignore:
            print('Skipping ID:', id)
        else:
            ids.append(id)
    return ids


def tile_ids_in(slide, level, root_dir, label, ignore=['.DS_Store','dict.csv', 'all.csv']):
    ids = []
    try:
        for id in os.listdir(root_dir):
            if id in ignore:
                print('Skipping ID:', id)
            else:
                ids.append([slide, level, root_dir+'/'+id, label])
    except FileNotFoundError:
        print('Ignore:', root_dir)

    return ids


# Get all svs images with its label as one file; level is the tile resolution level
def big_image_sum(path='../tiles/'):
    if not os.path.isdir(path):
        Cutter.cut()
    big_images = []
    for level in range(3):
        level = str(level)
        POLEimg = image_ids_in(path + "POLE/")
        SLimg = image_ids_in(path + "Serous-like/")
        MSIimg = image_ids_in(path + "MSI/")
        EMimg = image_ids_in(path + "Endometrioid/")
        for i in MSIimg:
            big_images.append([i, level, path + "MSI/{}/level{}".format(i, level), 0])
        for i in EMimg:
            big_images.append([i, level, path + "Endometrioid/{}/level{}".format(i, level), 1])
        for i in SLimg:
            big_images.append([i, level, path + "Serous-like/{}/level{}".format(i, level), 2])
        for i in POLEimg:
            big_images.append([i, level, path + "POLE/{}/level{}".format(i, level), 3])
    datapd = pd.DataFrame(big_images, columns=['slide', 'level', 'path', 'label'])
    datapd.to_csv(path + "All_images.csv", header=True, index=False)

    return datapd


# seperate into training and testing; each type is the same separation ratio on big images
# test and train csv files contain tiles' path.
def set_sep(alll, path, level=None, cut=0.2):
    trlist = []
    telist = []
    valist = []
    if level:
        alll = alll[alll.level == level]
    for i in range(4):
        subset = alll.loc[alll['label'] == i]
        unq = list(subset.slide.unique())
        np.random.shuffle(unq)
        validation = unq[:int(len(unq)*cut/4)]
        valist.append(subset[subset['slide'].isin(validation)])
        test = unq[int(len(unq)*cut/4):int(len(unq)*cut)]
        telist.append(subset[subset['slide'].isin(test)])
        train = unq[int(len(unq)*cut):]
        trlist.append(subset[subset['slide'].isin(train)])
    test = pd.concat(telist)
    train = pd.concat(trlist)
    validation = pd.concat(valist)
    test_tiles_list = []
    train_tiles_list = []
    validation_tiles_list = []
    for idx, row in test.iterrows():
        tile_ids = tile_ids_in(row['slide'], row['level'], row['path'], row['label'])
        test_tiles_list.extend(tile_ids)
    for idx, row in train.iterrows():
        tile_ids = tile_ids_in(row['slide'], row['level'], row['path'], row['label'])
        train_tiles_list.extend(tile_ids)
    for idx, row in validation.iterrows():
        tile_ids = tile_ids_in(row['slide'], row['level'], row['path'], row['label'])
        validation_tiles_list.extend(tile_ids)
    test_tiles = pd.DataFrame(test_tiles_list, columns=['slide', 'level', 'path', 'label'])
    train_tiles = pd.DataFrame(train_tiles_list, columns=['slide', 'level', 'path', 'label'])
    validation_tiles = pd.DataFrame(validation_tiles_list, columns=['slide', 'level', 'path', 'label'])
    # No shuffle on test set
    train_tiles = sku.shuffle(train_tiles)
    validation_tiles = sku.shuffle(validation_tiles)
    test_tiles.to_csv(path+'/te_sample.csv', header=True, index=False)
    train_tiles.to_csv(path+'/tr_sample.csv', header=True, index=False)
    validation_tiles.to_csv(path+'/va_sample.csv', header=True, index=False)

    return train_tiles, test_tiles, validation_tiles


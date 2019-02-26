"""
Prepare training and testing datasets as CSV dictionaries

Created on 11/26/2018

@author: RH
"""
import os
import pandas as pd
import sklearn.utils as sku
import numpy as np

tile_path = "../tiles/"


# get all full paths of images
def image_ids_in(root_dir, ignore=['.DS_Store','dict.csv','all.csv', 'tr_sample.csv', 'te_sample.csv']):
    ids = []
    for id in os.listdir(root_dir):
        if id in ignore:
            print('Skipping ID:', id)
        else:
            ids.append(id)
    return ids


def tile_ids_in(slide, root_dir, label, ignore=['.DS_Store','dict.csv','all.csv', 'tr_sample.csv', 'te_sample.csv']):
    ids = []
    for id in os.listdir(root_dir):
        if id in ignore:
            print('Skipping ID:', id)
        else:
            ids.append([slide, root_dir+'/'+id, label])
    return ids


# Get all svs images with its label as one file; level is the tile resolution level
def big_image_sum(level=None, path="../tiles/"):
    if level:
        level = str(level)
        big_images = []
        POLEimg = image_ids_in(path+"level{}/POLE/".format(level))
        SLimg = image_ids_in(path+"level{}/Serous-like/".format(level))
        MSIimg = image_ids_in(path+"level{}/MSI/".format(level))
        EMimg = image_ids_in(path+"level{}/Endometroid/".format(level))
        for i in MSIimg:
            big_images.append([i, path+"level{}/MSI/{}".format(level, i), 0])
        for i in EMimg:
            big_images.append([i, path+"level{}/Endometroid/{}".format(level, i), 1])
        for i in SLimg:
            big_images.append([i, path+"level{}/Serous-like/{}".format(level, i), 2])
        for i in POLEimg:
            big_images.append([i, path+"level{}/POLE/{}".format(level, i), 3])
        datapd = pd.DataFrame(big_images, columns=['slide', 'path', 'label'])
        datapd.to_csv(path+"level{}/All_images.csv".format(level), header=True, index=False)

    else:
        big_images = []
        for level in range(3):
            level = str(level)
            POLEimg = image_ids_in(path + "level{}/POLE/".format(level))
            SLimg = image_ids_in(path + "level{}/Serous-like/".format(level))
            MSIimg = image_ids_in(path + "level{}/MSI/".format(level))
            EMimg = image_ids_in(path + "level{}/Endometroid/".format(level))
            for i in MSIimg:
                big_images.append([i, path + "level{}/MSI/{}".format(level, i), 0])
            for i in EMimg:
                big_images.append([i, path + "level{}/Endometroid/{}".format(level, i), 1])
            for i in SLimg:
                big_images.append([i, path + "level{}/Serous-like/{}".format(level, i), 2])
            for i in POLEimg:
                big_images.append([i, path + "level{}/POLE/{}".format(level, i), 3])
        datapd = pd.DataFrame(big_images, columns=['slide', 'path', 'label'])
        datapd.to_csv(path + "All_images.csv", header=True, index=False)

    return datapd


# seperate into training and testing; each type is the same separation ratio on big images
# test and train csv files contain tiles' path.
def set_sep(alll, path, cut=0.2):
    trlist = []
    telist = []
    valist = []
    for i in range(4):
        subset = alll.loc[alll['label'] == i]
        subset = sku.shuffle(subset)
        validation, test, train = np.split(subset, [int(cut/4 * len(subset)), int(cut * len(subset))])
        trlist.append(train)
        telist.append(test)
        valist.append(validation)
    test = pd.concat(telist)
    train = pd.concat(trlist)
    validation = pd.concat(valist)
    test_tiles_list = []
    train_tiles_list = []
    validation_tiles_list = []
    for idx, row in test.iterrows():
        tile_ids = tile_ids_in(row['slide'], row['path'], row['label'])
        test_tiles_list.extend(tile_ids)
    for idx, row in train.iterrows():
        tile_ids = tile_ids_in(row['slide'], row['path'], row['label'])
        train_tiles_list.extend(tile_ids)
    for idx, row in validation.iterrows():
        tile_ids = tile_ids_in(row['slide'], row['path'], row['label'])
        validation_tiles_list.extend(tile_ids)
    test_tiles = pd.DataFrame(test_tiles_list, columns=['slide', 'path', 'label'])
    train_tiles = pd.DataFrame(train_tiles_list, columns=['slide', 'path', 'label'])
    validation_tiles = pd.DataFrame(validation_tiles_list, columns=['slide', 'path', 'label'])
    # No shuffle on test set
    train_tiles = sku.shuffle(train_tiles)
    validation_tiles = sku.shuffle(validation_tiles)
    test_tiles.to_csv(path+'/te_sample.csv', header=True, index=False)
    train_tiles.to_csv(path+'/tr_sample.csv', header=True, index=False)
    validation_tiles.to_csv(path+'/va_sample.csv', header=True, index=False)

    return train_tiles, test_tiles, validation_tiles


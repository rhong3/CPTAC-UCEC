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
def image_ids_in(root_dir, ignore=['.DS_Store','dict.csv', 'all.csv']):
    ids = []
    for id in os.listdir(root_dir):
        if id in ignore:
            print('Skipping ID:', id)
        else:
            ids.append(id)
    return ids


# Get intersection of 2 lists
def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


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
def big_image_sum(pmd, path='../tiles/', ref_file='../dummy_His_MUT_joined.csv'):
    if not os.path.isdir(path):
        os.mkdir(path)
        import Cutter
        Cutter.cut()
    allimg = image_ids_in(path)
    ref = pd.read_csv(ref_file, header=0)
    big_images = []
    for level in range(3):
        level = str(level)
        if pmd == 'subtype':
            ref = ref.loc[ref['subtype_0NA'] == 0]
            MSIimg = intersection(ref.loc[ref['subtype_MSI'] == 1]['name'].tolist(), allimg)
            EMimg = intersection(ref.loc[ref['subtype_Endometrioid'] == 1]['name'].tolist(), allimg)
            SLimg = intersection(ref.loc[ref['subtype_Serous-like'] == 1]['name'].tolist(), allimg)
            POLEimg = intersection(ref.loc[ref['subtype_POLE'] == 1]['name'].tolist(), allimg)
            for i in MSIimg:
                big_images.append([i, level, path + "{}/level{}".format(i, level), 0])
            for i in EMimg:
                big_images.append([i, level, path + "{}/level{}".format(i, level), 1])
            for i in SLimg:
                big_images.append([i, level, path + "{}/level{}".format(i, level), 2])
            for i in POLEimg:
                big_images.append([i, level, path + "{}/level{}".format(i, level), 3])
        elif pmd == 'histology':
            ref = ref.loc[ref['histology_0NA'] == 0]
            EMimg = intersection(ref.loc[ref['histology_Endometrioid'] == 1]['name'].tolist(), allimg)
            Serousimg = intersection(ref.loc[ref['histology_Serous'] == 1]['name'].tolist(), allimg)
            Mixedimg = intersection(ref.loc[ref['histology_Mixed'] == 1]['name'].tolist(), allimg)
            for i in EMimg:
                big_images.append([i, level, path + "{}/level{}".format(i, level), 0])
            for i in Serousimg:
                big_images.append([i, level, path + "{}/level{}".format(i, level), 1])
            for i in Mixedimg:
                big_images.append([i, level, path + "{}/level{}".format(i, level), 2])
        elif pmd in ['Endometrioid', 'MSI', 'Serous-like', 'POLE']:
            ref = ref.loc[ref['subtype_0NA'] == 0]
            negimg = intersection(ref.loc[ref['subtype_{}'.format(pmd)] == 0]['name'].tolist(), allimg)
            posimg = intersection(ref.loc[ref['subtype_{}'.format(pmd)] == 1]['name'].tolist(), allimg)
            for i in negimg:
                big_images.append([i, level, path + "{}/level{}".format(i, level), 0])
            for i in posimg:
                big_images.append([i, level, path + "{}/level{}".format(i, level), 1])
        elif pmd in ['histology_Endometrioid', 'histology_Serous', 'histology_Mixed']:
            ref = ref.loc[ref['histology_0NA'] == 0]
            negimg = intersection(ref.loc[ref[pmd] == 0]['name'].tolist(), allimg)
            posimg = intersection(ref.loc[ref[pmd] == 1]['name'].tolist(), allimg)
            for i in negimg:
                big_images.append([i, level, path + "{}/level{}".format(i, level), 0])
            for i in posimg:
                big_images.append([i, level, path + "{}/level{}".format(i, level), 1])
        else:
            negimg = intersection(ref.loc[ref[pmd] == 0]['name'].tolist(), allimg)
            posimg = intersection(ref.loc[ref[pmd] == 1]['name'].tolist(), allimg)
            for i in negimg:
                big_images.append([i, level, path + "{}/level{}".format(i, level), 0])
            for i in posimg:
                big_images.append([i, level, path + "{}/level{}".format(i, level), 1])

    datapd = pd.DataFrame(big_images, columns=['slide', 'level', 'path', 'label'])
    datapd.to_csv(path + "All_images.csv", header=True, index=False)

    return datapd


# seperate into training and testing; each type is the same separation ratio on big images
# test and train csv files contain tiles' path.
def set_sep(alll, path, cls, level=None, cut=0.2):
    trlist = []
    telist = []
    valist = []
    if level:
        alll = alll[alll.level == level]
    for i in range(cls):
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


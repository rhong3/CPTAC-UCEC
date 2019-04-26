"""
Prepare training and testing datasets as CSV dictionaries 2.0

Created on 04/26/2019

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


def tile_ids_in(slide, label, root_dir, ignore=['.DS_Store','dict.csv', 'all.csv']):
    dira = os.path.isdir(root_dir + 'level0')
    dirb = os.path.isdir(root_dir + 'level1')
    dirc = os.path.isdir(root_dir + 'level2')
    if dira and dirb and dirc:
        ids = []
        for level in range(3):
            level = str(level)
            dirr = root_dir + 'level{}'.format(level)
            for id in os.listdir(dirr):
                if id in ignore:
                    print('Skipping ID:', id)
                else:
                    ids.append([slide, label, level, root_dir+'/'+id])
        ids = pd.DataFrame(ids, columns=['slide', 'label', 'level', 'path'])
        idsa = ids.loc[ids['level'] == 0]
        idsb = ids.loc[ids['level'] == 1]
        idsc = ids.loc[ids['level'] == 2]
        idss = pd.DataFrame(columns=['slide', 'label', 'L0path', 'L1path', 'L2path'])
        idss['slide'] = idsa['slide']
        idss['label'] = idsa['label']
        idss['L0path'] = idsa['path']
        idss['L1path'] = idsb['path']
        idss['L2path'] = idsc['path']
        idss = sku.shuffle(idss)
        idss = idss.fillna(method='ffill')
        idss = idss.fillna(method='bfill')
    else:
        idss = pd.DataFrame(columns=['slide', 'label', 'L0path', 'L1path', 'L2path'])

    return idss


# Get all svs images with its label as one file; level is the tile resolution level
def big_image_sum(pmd, path='../tiles/', ref_file='../dummy_His_MUT_joined.csv'):
    if not os.path.isdir(path):
        os.mkdir(path)
        import Cutter
        Cutter.cut()
    allimg = image_ids_in(path)
    ref = pd.read_csv(ref_file, header=0)
    big_images = []
    if pmd == 'subtype':
        ref = ref.loc[ref['subtype_0NA'] == 0]
        MSIimg = intersection(ref.loc[ref['subtype_MSI'] == 1]['name'].tolist(), allimg)
        EMimg = intersection(ref.loc[ref['subtype_Endometrioid'] == 1]['name'].tolist(), allimg)
        SLimg = intersection(ref.loc[ref['subtype_Serous-like'] == 1]['name'].tolist(), allimg)
        POLEimg = intersection(ref.loc[ref['subtype_POLE'] == 1]['name'].tolist(), allimg)
        for i in MSIimg:
            big_images.append([i, 0,  path + "{}/".format(i)])
        for i in EMimg:
            big_images.append([i, 1, path + "{}/".format(i)])
        for i in SLimg:
            big_images.append([i, 2, path + "{}/".format(i)])
        for i in POLEimg:
            big_images.append([i, 3, path + "{}/".format(i)])
    elif pmd == 'histology':
        ref = ref.loc[ref['histology_0NA'] == 0]
        EMimg = intersection(ref.loc[ref['histology_Endometrioid'] == 1]['name'].tolist(), allimg)
        Serousimg = intersection(ref.loc[ref['histology_Serous'] == 1]['name'].tolist(), allimg)
        Mixedimg = intersection(ref.loc[ref['histology_Mixed'] == 1]['name'].tolist(), allimg)
        for i in EMimg:
            big_images.append([i, 0, path + "{}/".format(i)])
        for i in Serousimg:
            big_images.append([i, 1, path + "{}/".format(i)])
        for i in Mixedimg:
            big_images.append([i, 2, path + "{}/".format(i)])
    elif pmd in ['Endometrioid', 'MSI', 'Serous-like', 'POLE']:
        ref = ref.loc[ref['subtype_0NA'] == 0]
        negimg = intersection(ref.loc[ref['subtype_{}'.format(pmd)] == 0]['name'].tolist(), allimg)
        posimg = intersection(ref.loc[ref['subtype_{}'.format(pmd)] == 1]['name'].tolist(), allimg)
        for i in negimg:
            big_images.append([i, 0, path + "{}/".format(i)])
        for i in posimg:
            big_images.append([i, 1, path + "{}/".format(i)])
    elif pmd in ['histology_Endometrioid', 'histology_Serous', 'histology_Mixed']:
        ref = ref.loc[ref['histology_0NA'] == 0]
        negimg = intersection(ref.loc[ref[pmd] == 0]['name'].tolist(), allimg)
        posimg = intersection(ref.loc[ref[pmd] == 1]['name'].tolist(), allimg)
        for i in negimg:
            big_images.append([i, 0, path + "{}/".format(i)])
        for i in posimg:
            big_images.append([i, 1, path + "{}/".format(i)])
    else:
        negimg = intersection(ref.loc[ref[pmd] == 0]['name'].tolist(), allimg)
        posimg = intersection(ref.loc[ref[pmd] == 1]['name'].tolist(), allimg)
        for i in negimg:
            big_images.append([i, 0, path + "{}/".format(i)])
        for i in posimg:
            big_images.append([i, 1, path + "{}/".format(i)])

    datapd = pd.DataFrame(big_images, columns=['slide', 'label', 'path'])
    datapd.to_csv(path + "All_images.csv", header=True, index=False)

    return datapd


# seperate into training and testing; each type is the same separation ratio on big images
# test and train csv files contain tiles' path.
def set_sep(alll, path, cls, cut=0.2):
    trlist = []
    telist = []
    valist = []
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
    test_tiles = pd.DataFrame(columns=['slide', 'label', 'L0path', 'L1path', 'L2path'])
    train_tiles = pd.DataFrame(columns=['slide', 'label', 'L0path', 'L1path', 'L2path'])
    validation_tiles = pd.DataFrame(columns=['slide', 'label', 'L0path', 'L1path', 'L2path'])
    for idx, row in test.iterrows():
        tile_ids = tile_ids_in(row['slide'], row['label'], row['path'])
        test_tiles = pd.concat([test_tiles, tile_ids])
    for idx, row in train.iterrows():
        tile_ids = tile_ids_in(row['slide'], row['label'], row['path'])
        train_tiles = pd.concat([train_tiles, tile_ids])
    for idx, row in validation.iterrows():
        tile_ids = tile_ids_in(row['slide'], row['label'], row['path'])
        validation_tiles = pd.concat([validation_tiles, tile_ids])
    # No shuffle on test set
    train_tiles = sku.shuffle(train_tiles)
    validation_tiles = sku.shuffle(validation_tiles)
    # Use 30% of all tiles as we have tooooo many tiles
    train_tiles = train_tiles.sample(frac=0.3, replace=False)
    validation_tiles = validation_tiles.sample(frac=0.3, replace=False)
    test_tiles.to_csv(path+'/te_sample.csv', header=True, index=False)
    train_tiles.to_csv(path+'/tr_sample.csv', header=True, index=False)
    validation_tiles.to_csv(path+'/va_sample.csv', header=True, index=False)

    return train_tiles, test_tiles, validation_tiles


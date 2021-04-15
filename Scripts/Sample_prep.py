"""
Prepare training and testing datasets as CSV dictionaries

Created on 11/26/2018

@author: RH
"""
import os
import pandas as pd
import sklearn.utils as sku
import numpy as np


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


def tile_ids_in(slide, level, root_dir, label):
    ids = []
    try:
        for id in os.listdir(root_dir):
            if '.png' in id:
                ids.append([slide, level, root_dir+'/'+id, label])
            else:
                print('Skipping ID:', id)
    except FileNotFoundError:
        print('Ignore:', root_dir)

    return ids


# Balance CPTAC and TCGA tiles in each class
def balance(pdls, cls):
    balanced = pd.DataFrame(columns=['slide', 'level', 'path', 'label'])
    for i in range(cls):
        ref = pdls.loc[pdls['label'] == i]
        CPTAC = ref[~ref['slide'].str.contains("TCGA")]
        TCGA = ref[ref['slide'].str.contains("TCGA")]
        if CPTAC.shape[0] != 0 and TCGA.shape[0] != 0:
            ratio = (CPTAC.shape[0])/(TCGA.shape[0])
            if ratio < 0.2:
                TCGA = TCGA.sample(int(5*CPTAC.shape[0]), replace=False)
                ref = pd.concat([TCGA, CPTAC], sort=False)
            elif ratio > 5:
                CPTAC = CPTAC.sample(int(5*TCGA.shape[0]), replace=False)
                ref = pd.concat([TCGA, CPTAC], sort=False)
        balanced = pd.concat([balanced, ref], sort=False)
    return balanced


# Get all svs images with its label as one file; level is the tile resolution level
def big_image_sum(pmd, path='../tiles/', ref_file='../Fusion_dummy_His_MUT_joined.csv'):
    if not os.path.isdir(path):
        os.mkdir(path)
        import Cutter
        Cutter.cut()
    allimg = image_ids_in(path)
    ref = pd.read_csv(ref_file, header=0)
    big_images = []
    for level in range(4):
        level = str(level)
        if pmd == 'subtype':
            MSIimg = intersection(ref.loc[ref['subtype_MSI'] == 1]['name'].tolist(), allimg)
            EMimg = intersection(ref.loc[ref['subtype_Endometrioid'] == 1]['name'].tolist(), allimg)
            SLimg = intersection(ref.loc[ref['subtype_Serous-like'] == 1]['name'].tolist(), allimg)
            POLEimg = intersection(ref.loc[ref['subtype_POLE'] == 1]['name'].tolist(), allimg)
            for i in MSIimg:
                big_images.append([i, level, path + "{}/level{}".format(i, level), 1])
            for i in EMimg:
                big_images.append([i, level, path + "{}/level{}".format(i, level), 2])
            for i in SLimg:
                big_images.append([i, level, path + "{}/level{}".format(i, level), 3])
            for i in POLEimg:
                big_images.append([i, level, path + "{}/level{}".format(i, level), 0])
        elif pmd == 'histology':
            allimg = intersection(ref.loc[ref['histology_Mixed'] == 0]['name'].tolist(), allimg)
            EMimg = intersection(ref.loc[ref['histology_Endometrioid'] == 1]['name'].tolist(), allimg)
            Serousimg = intersection(ref.loc[ref['histology_Serous'] == 1]['name'].tolist(), allimg)
            for i in EMimg:
                big_images.append([i, level, path + "{}/level{}".format(i, level), 0])
            for i in Serousimg:
                big_images.append([i, level, path + "{}/level{}".format(i, level), 1])
        elif pmd in ['Endometrioid', 'MSI', 'Serous-like', 'POLE']:
            ref = ref.loc[ref['subtype_0NA'] == 0]

            ### special version
            # ref = ref.loc[ref['histology_Mixed'] == 0]
            # ref = ref.loc[ref['histology_Endometrioid'] == 1]
            ### special version

            negimg = intersection(ref.loc[ref['subtype_{}'.format(pmd)] == 0]['name'].tolist(), allimg)
            posimg = intersection(ref.loc[ref['subtype_{}'.format(pmd)] == 1]['name'].tolist(), allimg)
            for i in negimg:
                big_images.append([i, level, path + "{}/level{}".format(i, level), 0])
            for i in posimg:
                big_images.append([i, level, path + "{}/level{}".format(i, level), 1])
        elif pmd == 'MSIst':
            Himg = intersection(ref.loc[ref['MSIst_MSI-H'] == 1]['name'].tolist(), allimg)
            Simg = intersection(ref.loc[ref['MSIst_MSS'] == 1]['name'].tolist(), allimg)
            for i in Himg:
                big_images.append([i, level, path + "{}/level{}".format(i, level), 1])
            for i in Simg:
                big_images.append([i, level, path + "{}/level{}".format(i, level), 0])
        elif pmd in ['MSIst_MSI-H', 'MSIst_MSI-L', 'MSIst_MSS']:
            ref = ref.loc[ref['MSIst_0NA'] == 0]
            negimg = intersection(ref.loc[ref['subtype_{}'.format(pmd)] == 0]['name'].tolist(), allimg)
            posimg = intersection(ref.loc[ref['subtype_{}'.format(pmd)] == 1]['name'].tolist(), allimg)
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

    return datapd


# TO KEEP SPLIT SAME AS BASELINES. seperate into training and testing; each type is the same separation
# ratio on big images test and train csv files contain tiles' path.
def set_sep_secondary(alll, path, cls, pmd, level=None, batchsize=64):
    if level:
        alll = alll[alll.level == level]
    if pmd == 'subtype':
        split = pd.read_csv('../split/ST.csv', header=0)
    elif pmd == 'histology':
        split = pd.read_csv('../split/his.csv', header=0)
    elif pmd == 'Serous-like':
        split = pd.read_csv('../split/CNVH.csv', header=0)
    else:
        split = pd.read_csv('../split/{}.csv'.format(pmd), header=0)
    train = split.loc[split['set'] == 'train']['slide'].tolist()
    validation = split.loc[split['set'] == 'validation']['slide'].tolist()
    test = split.loc[split['set'] == 'test']['slide'].tolist()

    trlist = []
    telist = []
    valist = []

    subset = alll

    valist.append(subset[subset['slide'].isin(validation)])
    telist.append(subset[subset['slide'].isin(test)])
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
    train_tiles = balance(train_tiles, cls=cls)
    validation_tiles = balance(validation_tiles, cls=cls)
    # No shuffle on test set
    train_tiles = sku.shuffle(train_tiles)
    validation_tiles = sku.shuffle(validation_tiles)
    if train_tiles.shape[0] > int(batchsize * 80000 / 3):
        train_tiles = train_tiles.sample(int(batchsize * 80000 / 3), replace=False)
        print('Truncate training set!')
    if validation_tiles.shape[0] > int(batchsize * 80000 / 30):
        validation_tiles = validation_tiles.sample(int(batchsize * 80000 / 30), replace=False)
        print('Truncate validation set!')
    if test_tiles.shape[0] > int(batchsize * 80000 / 3):
        test_tiles = test_tiles.sample(int(batchsize * 80000 / 3), replace=False)
        print('Truncate test set!')

    test_tiles.to_csv(path + '/te_sample.csv', header=True, index=False)
    train_tiles.to_csv(path + '/tr_sample.csv', header=True, index=False)
    validation_tiles.to_csv(path + '/va_sample.csv', header=True, index=False)

    return train_tiles, test_tiles, validation_tiles


# Training and validation on TCGA; Testing on CPTAC
def set_sep_idp(alll, path, cls, level=None, cut=0.1, batchsize=64):
    trlist = []
    telist = []
    valist = []
    if level:
        alll = alll[alll.level == level]

    TCGA = alll[alll['slide'].str.contains("TCGA")]
    CPTAC = alll[~alll['slide'].str.contains("TCGA")]
    for i in range(cls):
        subset = TCGA.loc[TCGA['label'] == i]
        unq = list(subset.slide.unique())
        np.random.shuffle(unq)
        validation = unq[:int(len(unq) * cut)]
        valist.append(subset[subset['slide'].isin(validation)])
        train = unq[int(len(unq) * cut):]
        trlist.append(subset[subset['slide'].isin(train)])
    telist.append(CPTAC)
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
    if train_tiles.shape[0] > int(batchsize * 30000):
        train_tiles = train_tiles.sample(int(batchsize * 30000), replace=False)
        print('Truncate training set!')
    if validation_tiles.shape[0] > int(batchsize * 3000):
        validation_tiles = validation_tiles.sample(int(batchsize * 3000), replace=False)
        print('Truncate validation set!')

    test_tiles.to_csv(path + '/te_sample.csv', header=True, index=False)
    train_tiles.to_csv(path + '/tr_sample.csv', header=True, index=False)
    validation_tiles.to_csv(path + '/va_sample.csv', header=True, index=False)

    return train_tiles, test_tiles, validation_tiles


# seperate into training and testing; each type is the same separation ratio on big images
# test and train csv files contain tiles' path.
def set_sep(alll, path, cls, level=None, cut=0.3, batchsize=64):
    trlist = []
    telist = []
    valist = []
    if level:
        alll = alll[alll.level == level]

    TCGA = alll[alll['slide'].str.contains("TCGA")]
    CPTAC = alll[~alll['slide'].str.contains("TCGA")]
    for i in range(cls):
        subset = TCGA.loc[TCGA['label'] == i]
        unq = list(subset.slide.unique())
        np.random.shuffle(unq)
        validation = unq[:int(len(unq) * cut / 2)]
        valist.append(subset[subset['slide'].isin(validation)])
        test = unq[int(len(unq) * cut / 2):int(len(unq) * cut)]
        telist.append(subset[subset['slide'].isin(test)])
        train = unq[int(len(unq) * cut):]
        trlist.append(subset[subset['slide'].isin(train)])

        subset = CPTAC.loc[CPTAC['label'] == i]
        unq = list(subset.slide.unique())
        np.random.shuffle(unq)
        validation = unq[:int(len(unq) * cut / 2)]
        valist.append(subset[subset['slide'].isin(validation)])
        test = unq[int(len(unq) * cut / 2):int(len(unq) * cut)]
        telist.append(subset[subset['slide'].isin(test)])
        train = unq[int(len(unq) * cut):]
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

    train_tiles = balance(train_tiles, cls=cls)
    validation_tiles = balance(validation_tiles, cls=cls)
    # No shuffle on test set
    train_tiles = sku.shuffle(train_tiles)
    validation_tiles = sku.shuffle(validation_tiles)
    if train_tiles.shape[0] > int(batchsize*30000):
        train_tiles = train_tiles.sample(int(batchsize*30000), replace=False)
        print('Truncate training set!')
    if validation_tiles.shape[0] > int(batchsize*3000):
        validation_tiles = validation_tiles.sample(int(batchsize*3000), replace=False)
        print('Truncate validation set!')
    if test_tiles.shape[0] > int(batchsize*30000):
        test_tiles = test_tiles.sample(int(batchsize*30000), replace=False)
        print('Truncate test set!')

    test_tiles.to_csv(path+'/te_sample.csv', header=True, index=False)
    train_tiles.to_csv(path+'/tr_sample.csv', header=True, index=False)
    validation_tiles.to_csv(path+'/va_sample.csv', header=True, index=False)

    return train_tiles, test_tiles, validation_tiles


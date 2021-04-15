"""
Prepare training and testing datasets as CSV dictionaries 2.0

Created on 04/26/2019; modified on 11/06/2019

@author: RH
"""
import os
import pandas as pd
import sklearn.utils as sku
import numpy as np
import re


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


# pair tiles of 20x, 10x, 5x of the same area
def paired_tile_ids_in_old(slide, label, root_dir):
    dira = os.path.isdir(root_dir + 'level0')
    dirb = os.path.isdir(root_dir + 'level1')
    dirc = os.path.isdir(root_dir + 'level2')
    if dira and dirb and dirc:
        if "TCGA" in root_dir:
            fac = 1000
        else:
            fac = 500
        ids = []
        for level in range(3):
            dirr = root_dir + 'level{}'.format(str(level))
            for id in os.listdir(dirr):
                if '.png' in id:
                    x = int(float(id.split('x-', 1)[1].split('-', 1)[0]) / fac)
                    y = int(float(re.split('_', id.split('y-', 1)[1])[0]) / fac)
                    try:
                        dup = re.split('.p', re.split('_', id.split('y-', 1)[1])[1])[0]
                    except IndexError:
                        dup = np.nan
                    ids.append([slide, label, level, dirr + '/' + id, x, y, dup])
                else:
                    print('Skipping ID:', id)
        ids = pd.DataFrame(ids, columns=['slide', 'label', 'level', 'path', 'x', 'y', 'dup'])
        idsa = ids.loc[ids['level'] == 0]
        idsa = idsa.drop(columns=['level'])
        idsa = idsa.rename(index=str, columns={"path": "L0path"})
        idsb = ids.loc[ids['level'] == 1]
        idsb = idsb.drop(columns=['slide', 'label', 'level'])
        idsb = idsb.rename(index=str, columns={"path": "L1path"})
        idsc = ids.loc[ids['level'] == 2]
        idsc = idsc.drop(columns=['slide', 'label', 'level'])
        idsc = idsc.rename(index=str, columns={"path": "L2path"})
        idsa = pd.merge(idsa, idsb, on=['x', 'y', 'dup'], how='left', validate="many_to_many")
        idsa['x'] = idsa['x'] - (idsa['x'] % 2)
        idsa['y'] = idsa['y'] - (idsa['y'] % 2)
        idsa = pd.merge(idsa, idsc, on=['x', 'y', 'dup'], how='left', validate="many_to_many")
        idsa = idsa.drop(columns=['x', 'y', 'dup'])
        idsa = idsa.dropna()
        idsa = sku.shuffle(idsa)
    else:
        idsa = pd.DataFrame(columns=['slide', 'label', 'L0path', 'L1path', 'L2path'])

    return idsa


def tile_ids_in(inp):
    ids = []
    try:
        for id in os.listdir(inp['path']):
            if '_{}.png'.format(str(inp['sldnum'])) in id:
                ids.append([inp['slide'], inp['level'], inp['path']+'/'+id, inp['BMI'], inp['age'], inp['label']])
    except FileNotFoundError:
        print('Ignore:', inp['path'])

    return ids


# pair tiles of 10x, 5x, 2.5x of the same area
def paired_tile_ids_in(slide, label, root_dir, age=None, BMI=None):
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
                    ids.append([slide, label, level, dirr + '/' + id, x, y, dup])
                else:
                    print('Skipping ID:', id)
        ids = pd.DataFrame(ids, columns=['slide', 'label', 'level', 'path', 'x', 'y', 'dup'])
        idsa = ids.loc[ids['level'] == 1]
        idsa = idsa.drop(columns=['level'])
        idsa = idsa.rename(index=str, columns={"path": "L0path"})
        idsb = ids.loc[ids['level'] == 2]
        idsb = idsb.drop(columns=['slide', 'label', 'level'])
        idsb = idsb.rename(index=str, columns={"path": "L1path"})
        idsc = ids.loc[ids['level'] == 3]
        idsc = idsc.drop(columns=['slide', 'label', 'level'])
        idsc = idsc.rename(index=str, columns={"path": "L2path"})
        idsa = pd.merge(idsa, idsb, on=['x', 'y', 'dup'], how='left', validate="many_to_many")
        idsa['x'] = idsa['x'] - (idsa['x'] % 2)
        idsa['y'] = idsa['y'] - (idsa['y'] % 2)
        idsa = pd.merge(idsa, idsc, on=['x', 'y', 'dup'], how='left', validate="many_to_many")
        idsa = idsa.drop(columns=['x', 'y', 'dup'])
        idsa = idsa.dropna()
        idsa = sku.shuffle(idsa)
        idsa['age'] = age
        idsa['BMI'] = BMI
    else:
        idsa = pd.DataFrame(columns=['slide', 'label', 'L0path', 'L1path', 'L2path', 'age', 'BMI'])

    return idsa


# Balance CPTAC and TCGA tiles in each class
def balance(pdls, cls):
    balanced = pd.DataFrame(columns=['slide', 'label', 'L0path', 'L1path', 'L2path', 'age', 'BMI'])
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


# Prepare label at per patient level
def big_image_sum(pmd, path='../tiles/', ref_file='../Fusion_dummy_His_MUT_joined.csv'):
    ref = pd.read_csv(ref_file, header=0)
    big_images = []
    if pmd == 'subtype':
        ref = ref.loc[ref['subtype_0NA'] == 0]
        for idx, row in ref.iterrows():
            if row['subtype_POLE'] == 1:
                big_images.append([row['name'], 0, path + "{}/".format(str(row['name'])), row['age'], row['BMI']])
            elif row['subtype_MSI'] == 1:
                big_images.append([row['name'], 1, path + "{}/".format(str(row['name'])), row['age'], row['BMI']])
            elif row['subtype_Endometrioid'] == 1:
                big_images.append([row['name'], 2, path + "{}/".format(str(row['name'])), row['age'], row['BMI']])
            elif row['subtype_Serous-like'] == 1:
                big_images.append([row['name'], 3, path + "{}/".format(str(row['name'])), row['age'], row['BMI']])
    elif pmd == 'histology':
        ref = ref.loc[ref['histology_Mixed'] == 0]
        for idx, row in ref.iterrows():
            if row['histology_Endometrioid'] == 1:
                big_images.append([row['name'], 0, path + "{}/".format(str(row['name'])), row['age'], row['BMI']])
            if row['histology_Serous'] == 1:
                big_images.append([row['name'], 1, path + "{}/".format(str(row['name'])), row['age'], row['BMI']])
    elif pmd in ['Endometrioid', 'MSI', 'Serous-like', 'POLE']:
        # ref = ref.loc[ref['histology_Endometrioid'] == 1]
        ref = ref.loc[ref['subtype_0NA'] == 0]
        for idx, row in ref.iterrows():
            big_images.append([row['name'], int(row['subtype_{}'.format(pmd)]), path + "{}/".format(str(row['name'])),
                               row['age'], row['BMI']])
    elif pmd == 'MSIst':
        ref = ref.loc[ref['MSIst_0NA'] == 0]
        for idx, row in ref.iterrows():
            big_images.append([row['name'], int(row['MSIst_MSI-H']), path + "{}/".format(str(row['name'])),
                               row['age'], row['BMI']])
    else:
        ref = ref.dropna(subset=[pmd])
        for idx, row in ref.iterrows():
            big_images.append([row['name'], int(row[pmd]), path + "{}/".format(str(row['name'])), row['age'], row['BMI']])

    datapd = pd.DataFrame(big_images, columns=['slide', 'label', 'path', 'age', 'BMI'])
    datapd = datapd.dropna()

    return datapd


# TO KEEP SPLIT SAME AS BASELINES. seperate into training and testing; each type is the same separation
# ratio on big images test and train csv files contain tiles' path.
def set_sep_secondary(alll, path, cls, pmd, batchsize=24):
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

    test_tiles = pd.DataFrame(columns=['slide', 'label', 'L0path', 'L1path', 'L2path', 'age', 'BMI'])
    train_tiles = pd.DataFrame(columns=['slide', 'label', 'L0path', 'L1path', 'L2path', 'age', 'BMI'])
    validation_tiles = pd.DataFrame(columns=['slide', 'label', 'L0path', 'L1path', 'L2path', 'age', 'BMI'])
    for idx, row in test.iterrows():
        tile_ids = paired_tile_ids_in(row['slide'], row['label'], row['path'], row['age'], row['BMI'])
        test_tiles = pd.concat([test_tiles, tile_ids])
    for idx, row in train.iterrows():
        tile_ids = paired_tile_ids_in(row['slide'], row['label'], row['path'], row['age'], row['BMI'])
        train_tiles = pd.concat([train_tiles, tile_ids])
    for idx, row in validation.iterrows():
        tile_ids = paired_tile_ids_in(row['slide'], row['label'], row['path'], row['age'], row['BMI'])
        validation_tiles = pd.concat([validation_tiles, tile_ids])

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
def set_sep_idp(alll, path, cls, cut=0.1, batchsize=64):
    trlist = []
    telist = []
    valist = []

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
    test_tiles = pd.DataFrame(columns=['slide', 'label', 'L0path', 'L1path', 'L2path', 'age', 'BMI'])
    train_tiles = pd.DataFrame(columns=['slide', 'label', 'L0path', 'L1path', 'L2path', 'age', 'BMI'])
    validation_tiles = pd.DataFrame(columns=['slide', 'label', 'L0path', 'L1path', 'L2path', 'age', 'BMI'])
    for idx, row in test.iterrows():
        tile_ids = paired_tile_ids_in(row['slide'], row['label'], row['path'], row['age'], row['BMI'])
        test_tiles = pd.concat([test_tiles, tile_ids])
    for idx, row in train.iterrows():
        tile_ids = paired_tile_ids_in(row['slide'], row['label'], row['path'], row['age'], row['BMI'])
        train_tiles = pd.concat([train_tiles, tile_ids])
    for idx, row in validation.iterrows():
        tile_ids = paired_tile_ids_in(row['slide'], row['label'], row['path'], row['age'], row['BMI'])
        validation_tiles = pd.concat([validation_tiles, tile_ids])

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


# seperate into training and testing; each type is the same separation ratio on big images
# test and train csv files contain tiles' path.
def set_sep(alll, path, cls, cut=0.2, batchsize=24):
    trlist = []
    telist = []
    valist = []
    TCGA = alll[alll['slide'].str.contains("TCGA")]
    CPTAC = alll[~alll['slide'].str.contains("TCGA")]
    for i in range(cls):
        subset = TCGA.loc[TCGA['label'] == i]
        unq = list(subset.slide.unique())
        np.random.shuffle(unq)
        validation = unq[:int(len(unq)*cut/2)]
        valist.append(subset[subset['slide'].isin(validation)])
        test = unq[int(len(unq)*cut/2):int(len(unq)*cut)]
        telist.append(subset[subset['slide'].isin(test)])
        train = unq[int(len(unq)*cut):]
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

    test_tiles = pd.DataFrame(columns=['slide', 'label', 'L0path', 'L1path', 'L2path', 'age', 'BMI'])
    train_tiles = pd.DataFrame(columns=['slide', 'label', 'L0path', 'L1path', 'L2path', 'age', 'BMI'])
    validation_tiles = pd.DataFrame(columns=['slide', 'label', 'L0path', 'L1path', 'L2path', 'age', 'BMI'])
    for idx, row in test.iterrows():
        tile_ids = paired_tile_ids_in(row['slide'], row['label'], row['path'], row['age'], row['BMI'])
        test_tiles = pd.concat([test_tiles, tile_ids])
    for idx, row in train.iterrows():
        tile_ids = paired_tile_ids_in(row['slide'], row['label'], row['path'], row['age'], row['BMI'])
        train_tiles = pd.concat([train_tiles, tile_ids])
    for idx, row in validation.iterrows():
        tile_ids = paired_tile_ids_in(row['slide'], row['label'], row['path'], row['age'], row['BMI'])
        validation_tiles = pd.concat([validation_tiles, tile_ids])

    train_tiles = balance(train_tiles, cls=cls)
    validation_tiles = balance(validation_tiles, cls=cls)
    # No shuffle on test set
    train_tiles = sku.shuffle(train_tiles)
    validation_tiles = sku.shuffle(validation_tiles)
    if train_tiles.shape[0] > int(batchsize*80000/3):
        train_tiles = train_tiles.sample(int(batchsize*80000/3), replace=False)
        print('Truncate training set!')
    if validation_tiles.shape[0] > int(batchsize*80000/30):
        validation_tiles = validation_tiles.sample(int(batchsize*80000/30), replace=False)
        print('Truncate validation set!')
    if test_tiles.shape[0] > int(batchsize*80000/3):
        test_tiles = test_tiles.sample(int(batchsize*80000/3), replace=False)
        print('Truncate test set!')

    test_tiles.to_csv(path+'/te_sample.csv', header=True, index=False)
    train_tiles.to_csv(path+'/tr_sample.csv', header=True, index=False)
    validation_tiles.to_csv(path+'/va_sample.csv', header=True, index=False)

    return train_tiles, test_tiles, validation_tiles

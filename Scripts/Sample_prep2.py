"""
Prepare training and testing datasets as CSV dictionaries 2.0

Created on 04/26/2019

@author: RH
"""
import os
import pandas as pd
import sklearn.utils as sku
import numpy as np
import re

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


def tile_ids_in(slide, label, root_dir):
    dira = os.path.isdir(root_dir + 'level0')
    dirb = os.path.isdir(root_dir + 'level1')
    dirc = os.path.isdir(root_dir + 'level2')
    if dira and dirb and dirc:
        ids = []
        for level in range(3):
            dirr = root_dir + 'level{}'.format(str(level))
            for id in os.listdir(dirr):
                if '.png' in id:
                    ids.append([slide, label, level, dirr + '/' + id])
                else:
                    print('Skipping ID:', id)
        ids = pd.DataFrame(ids, columns=['slide', 'label', 'level', 'path'])
        idsa = ids.loc[ids['level'] == 0]
        idsb = ids.loc[ids['level'] == 1]
        idsc = ids.loc[ids['level'] == 2]
        idss = pd.DataFrame(columns=['slide', 'label', 'L0path', 'L1path', 'L2path'])
        idss['slide'] = idsa['slide']
        idss['label'] = idsa['label']
        idss['L0path'] = idsa['path']
        idss = idss.reset_index(drop=True)
        idsb = idsb.reset_index(drop=True)
        idsc = idsc.reset_index(drop=True)
        idss['L1path'] = idsb['path']
        idss['L2path'] = idsc['path']
        idss = sku.shuffle(idss)
        idss = idss.fillna(method='ffill')
        idss = idss.fillna(method='bfill')
    else:
        idss = pd.DataFrame(columns=['slide', 'label', 'L0path', 'L1path', 'L2path'])

    return idss


# pair tiles of 20x, 10x, 5x of the same area
def paired_tile_ids_in(slide, label, root_dir):
    dira = os.path.isdir(root_dir + 'level0')
    dirb = os.path.isdir(root_dir + 'level1')
    dirc = os.path.isdir(root_dir + 'level2')
    if dira and dirb and dirc:
        if "TCGA" in root_dir:
            fac = 2000.1
        else:
            fac = 1000.1
        ids = []
        for level in range(3):
            dirr = root_dir + 'level{}'.format(str(level))
            for id in os.listdir(dirr):
                if '.png' in id:
                    x = int(float(id.split('x-', 1)[1].split('-', 1)[0]) / fac)
                    y = int(float(re.split('.p| |_', id.split('y-', 1)[1])[0]) / fac)
                    try:
                        dup = int(re.split('.p', re.split('_', id.split('y-', 1)[1])[1])[0])
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
        idsa = pd.merge(idsa, idsc, on=['x', 'y', 'dup'], how='left', validate="many_to_many")
        idsa = idsa.drop(columns=['x', 'y', 'dup'])
        # idsa = idsa.fillna(method='ffill', axis=0)
        # idsa = idsa.fillna(method='bfill', axis=0)
        idsa = idsa.dropna()
        idsa = sku.shuffle(idsa)
    else:
        idsa = pd.DataFrame(columns=['slide', 'label', 'L0path', 'L1path', 'L2path'])

    return idsa


# Balance CPTAC and TCGA tiles in each class
def balance(pdls, cls):
    balanced = pd.DataFrame(columns=['slide', 'label', 'L0path', 'L1path', 'L2path'])
    for i in range(cls):
        ref = pdls.loc[pdls['label'] == i]
        CPTAC = ref[~ref['slide'].str.contains("TCGA")]
        TCGA = ref[ref['slide'].str.contains("TCGA")]
        if CPTAC.shape[0] != 0 and TCGA.shape[0] != 0:
            ratio = (CPTAC.shape[0])/(TCGA.shape[0])
            if ratio < 0.25:
                TCGA = TCGA.sample(int(4*CPTAC.shape[0]), replace=False)
                ref = pd.concat([TCGA, CPTAC], sort=False)
            elif ratio > 4:
                CPTAC = CPTAC.sample(int(4*TCGA.shape[0]), replace=False)
                ref = pd.concat([TCGA, CPTAC], sort=False)
        balanced = pd.concat([balanced, ref], sort=False)
    return balanced


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
        EMimg = intersection(ref.loc[ref['histology_Endometrioid'] == 1]['name'].tolist(), allimg)
        Serousimg = intersection(ref.loc[ref['histology_Serous'] == 1]['name'].tolist(), allimg)
        for i in EMimg:
            big_images.append([i, 0, path + "{}/".format(i)])
        for i in Serousimg:
            big_images.append([i, 1, path + "{}/".format(i)])
    elif pmd in ['Endometrioid', 'MSI', 'Serous-like', 'POLE']:
        ref = ref.loc[ref['subtype_0NA'] == 0]
        negimg = intersection(ref.loc[ref['subtype_{}'.format(pmd)] == 0]['name'].tolist(), allimg)
        posimg = intersection(ref.loc[ref['subtype_{}'.format(pmd)] == 1]['name'].tolist(), allimg)
        for i in negimg:
            big_images.append([i, 0, path + "{}/".format(i)])
        for i in posimg:
            big_images.append([i, 1, path + "{}/".format(i)])
    elif pmd == 'MSIst':
        Himg = intersection(ref.loc[ref['MSIst_MSI-H'] == 1]['name'].tolist(), allimg)
        Simg = intersection(ref.loc[ref['MSIst_MSS'] == 1]['name'].tolist(), allimg)
        for i in Himg:
            big_images.append([i, 0, path + "{}/".format(i)])
        for i in Simg:
            big_images.append([i, 1, path + "{}/".format(i)])
    elif pmd in ['MSIst_MSI-H', 'MSIst_MSI-L', 'MSIst_MSS']:
        ref = ref.loc[ref['MSIst_0NA'] == 0]
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

    return datapd


# seperate into training and testing; each type is the same separation ratio on big images
# test and train csv files contain tiles' path.
def set_sep(alll, path, cls, cut=0.2):
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

    test_tiles = pd.DataFrame(columns=['slide', 'label', 'L0path', 'L1path', 'L2path'])
    train_tiles = pd.DataFrame(columns=['slide', 'label', 'L0path', 'L1path', 'L2path'])
    validation_tiles = pd.DataFrame(columns=['slide', 'label', 'L0path', 'L1path', 'L2path'])
    for idx, row in test.iterrows():
        tile_ids = paired_tile_ids_in(row['slide'], row['label'], row['path'])
        test_tiles = pd.concat([test_tiles, tile_ids])
    for idx, row in train.iterrows():
        tile_ids = paired_tile_ids_in(row['slide'], row['label'], row['path'])
        train_tiles = pd.concat([train_tiles, tile_ids])
    for idx, row in validation.iterrows():
        tile_ids = paired_tile_ids_in(row['slide'], row['label'], row['path'])
        validation_tiles = pd.concat([validation_tiles, tile_ids])

    train_tiles = train_tiles.sample(frac=0.2, replace=False)
    validation_tiles = validation_tiles.sample(frac=0.2, replace=False)
    test_tiles = test_tiles.sample(frac=0.2, replace=False)
    train_tiles = balance(train_tiles, cls=cls)
    validation_tiles = balance(validation_tiles, cls=cls)
    # No shuffle on test set
    # train_tiles = sku.shuffle(train_tiles)
    # validation_tiles = sku.shuffle(validation_tiles)

    test_tiles.to_csv(path+'/te_sample.csv', header=True, index=False)
    train_tiles.to_csv(path+'/tr_sample.csv', header=True, index=False)
    validation_tiles.to_csv(path+'/va_sample.csv', header=True, index=False)

    return train_tiles, test_tiles, validation_tiles


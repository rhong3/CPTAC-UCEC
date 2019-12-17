"""
Count all slides size

Created on 12/17/2019

@author: RH
"""
from openslide import OpenSlide
import numpy as np
import pandas as pd
import os
import re


# Get all images in the root directory
def image_ids_in(root_dir, mode, ignore=['.DS_Store', 'dict.csv']):
    ids = []
    for id in os.listdir(root_dir):
        if id in ignore:
            print('Skipping ID:', id)
        else:
            if mode == 'CPTAC':
                dirname = id.split('_')[-3]
                sldname = id.split('_')[-2]
                sldnum = id.split('_')[-2].split('-')[-1]
                ids.append((id, dirname, sldname, sldnum))
            if mode == 'TCGA':
                dirname = re.split('-01Z|-02Z', id)[0]
                sldnum = id.split('-')[5].split('.')[0]
                sldname = id.split('.')[0]
                ids.append((id, dirname, sldname, sldnum))
    return ids


def count_tile(root_dir, sld):
    ct = 0
    try:
        for id in os.listdir(root_dir):
            if sld in id and '.png' in id:
                ct += 1
    except Exception as e:
        print('Error! '+root_dir)
    return ct


def cut():
    CPTACpath = '../images/CPTAC/'
    TCGApath = '../images/TCGA/'
    ref = pd.read_csv('../dummy_His_MUT_joined.csv', header=0)
    # cut tiles with coordinates in the name (exclude white)
    CPTAClist = image_ids_in(CPTACpath, 'CPTAC')
    TCGAlist = image_ids_in(TCGApath, 'TCGA')

    # CPTAC
    CPTAClist_new = []
    for i in CPTAClist:
        matchrow = ref.loc[ref['name'] == i[1]]
        if matchrow.empty:
            continue
        slide = OpenSlide('../images/CPTAC/'+i[0])
        bounds_width = slide.level_dimensions[0][0]
        bounds_height = slide.level_dimensions[0][1]
        j = i + (bounds_width, bounds_height)
        for f in range(4):
            tl_num = count_tile('../tiles/{}/level{}'.format(i[2], str(f)), str('_'+i[3]))
            j = j + (tl_num, )

        CPTAClist_new.append(j)
    CPTACpp = pd.DataFrame(CPTAClist_new, columns=['id', 'dir', 'sld', 'sld_num', 'width',
                                                   'height', '20Xtiles', '10Xtiles', '5Xtiles', '2.5Xtiles'])

    # TCGA
    TCGAlist_new = []
    for i in TCGAlist:
        matchrow = ref.loc[ref['name'] == i[1]]
        if matchrow.empty:
            continue
        slide = OpenSlide('../images/TCGA/' + i[0])
        bounds_width = slide.level_dimensions[0][0]
        bounds_height = slide.level_dimensions[0][1]
        j = i + (bounds_width, bounds_height)
        for f in range(4):
            tl_num = count_tile('../tiles/{}/level{}'.format(i[2], str(f)), str('_'+i[3]))
            j = j + (tl_num, )
        TCGAlist_new.append(j)
    TCGApp = pd.DataFrame(TCGAlist_new, columns=['id', 'dir', 'sld', 'sld_num', 'width',
                                                   'height', '20Xtiles', '10Xtiles', '5Xtiles', '2.5Xtiles'])

    result = pd.concat([CPTACpp, TCGApp], axis=0, sort=False)

    result.to_csv('../Results/slide_dim.csv', index=False)


# Run as main
if __name__ == "__main__":
    cut()

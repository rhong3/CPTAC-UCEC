"""
Tile svs/scn files

Created on 11/01/2018

@author: RH
"""

import time
import matplotlib
import os
import shutil
import pandas as pd
import numpy as np
matplotlib.use('Agg')
import Slicer
import staintools


# Get all images in the root directory
def image_ids_in(root_dir, mode, ignore=['.DS_Store', 'dict.csv']):
    ids = []
    for id in os.listdir(root_dir):
        if id in ignore:
            print('Skipping ID:', id)
        else:
            if mode == 'CPTAC':
                dirname = id.split('_')[-3]
                sldnum = id.split('_')[-2].split('-')[-1]
                ids.append((id, dirname, sldnum))
            if mode == 'TCGA':
                dirname = id.split('-01Z')[0]
                sldnum = id.split('-')[5].split('.')[0]
                ids.append((id, dirname, sldnum))
    return ids


# cut; each level is 2 times difference (20x, 10x, 5x)
def cut():
    # load standard image for normalization
    std = staintools.read_image("../colorstandard.png")
    std = staintools.LuminosityStandardizer.standardize(std)
    CPTACpath = '../images/CPTAC/'
    TCGApath = '../images/TCGA/'
    ref = pd.read_csv('../dummy_His_MUT_joined.csv', header=0)

    # cut tiles with coordinates in the name (exclude white)
    start_time = time.time()
    CPTAClist = image_ids_in(CPTACpath, 'CPTAC')
    TCGAlist = image_ids_in(TCGApath, 'TCGA')

    CPTACpp = pd.DataFrame(CPTAClist, columns=['id', 'dir', 'sld'])
    CPTACcc = CPTACpp['dir'].value_counts()
    CPTACcc = CPTACcc[CPTACcc > 1].index.tolist()
    print(CPTACcc)

    TCGApp = pd.DataFrame(TCGAlist, columns=['id', 'dir', 'sld'])
    TCGAcc = TCGApp['dir'].value_counts()
    TCGAcc = TCGAcc[TCGAcc > 1].index.tolist()
    print(TCGAcc)

    # CPTAC
    for i in CPTAClist:
        matchrow = ref.loc[ref['name'] == i[1]]
        if matchrow.empty:
            continue
        # if i[1] in CPTACcc:
        try:
            os.mkdir("../tiles/{}".format(i[1]))
        except(FileExistsError):
            pass
        for m in range(3, 4):
            if m == 0:
                tff = 1
                level = 0
            elif m == 1:
                tff = 2
                level = 0
            elif m == 2:
                tff = 1
                level = 1
            elif m == 3:
                tff = 2
                level = 1
            otdir = "../tiles/{}/level{}".format(i[1], str(m))
            try:
                os.mkdir(otdir)
            except(FileExistsError):
                pass
            try:
                n_x, n_y, raw_img, resx, resy, ct = Slicer.tile(image_file='CPTAC/'+i[0], outdir=otdir,
                                                                level=level, std_img=std, dp=i[2], ft=tff)
            except(IndexError):
                pass
            if len(os.listdir(otdir)) < 2:
                shutil.rmtree(otdir, ignore_errors=True)
        # else:
        #     print("pass: {}".format(str(i)))

    # TCGA
    for i in TCGAlist:
        matchrow = ref.loc[ref['name'] == i[1]]
        if matchrow.empty:
            continue
        # if i[1] in TCGAcc:
        try:
            os.mkdir("../tiles/{}".format(i[1]))
        except(FileExistsError):
            pass
        for m in range(3, 4):
            if m == 0:
                tff = 2
                level = 0
            elif m == 1:
                tff = 1
                level = 1
            elif m == 2:
                tff = 2
                level = 1
            elif m == 3:
                tff = 1
                level = 2
            otdir = "../tiles/{}/level{}".format(i[1], str(m))
            try:
                os.mkdir(otdir)
            except(FileExistsError):
                pass
            try:
                n_x, n_y, raw_img, resx, resy, ct = Slicer.tile(image_file='TCGA/'+i[0], outdir=otdir,
                                                                level=level, std_img=std, dp=i[2], ft=tff)
            except Exception as e:
                print('Error!')
                pass

            if len(os.listdir(otdir)) < 2:
                shutil.rmtree(otdir, ignore_errors=True)
        # else:
        #     print("pass: {}".format(str(i)))

    print("--- %s seconds ---" % (time.time() - start_time))

    # # Time measure tool
    # start_time = time.time()
    # print("--- %s seconds ---" % (time.time() - start_time))


# Run as main
if __name__ == "__main__":
    if not os.path.isdir('../tiles'):
        os.mkdir('../tiles')
    cut()


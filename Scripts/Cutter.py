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


# Get all images in the root directory
def image_ids_in(root_dir, mode, ignore=['.DS_Store', 'dict.csv']):
    ids = []
    for id in os.listdir(root_dir):
        if id in ignore:
            print('Skipping ID:', id)
        else:
            if mode == 'CPTAC':
                dirname = id.split('_')[-3]
                ids.append((id, dirname))
            if mode == 'TCGA':
                dirname = id.split('-01Z')[0]
                ids.append((id, dirname))
    return ids


def cut():
    CPTACpath = '../images/CPTAC/'
    TCGApath = '../images/TCGA/'
    ref = pd.read_csv('../dummy_MUT_joined.csv', header=0)

    # cut tiles with coordinates in the name (exclude white)
    start_time = time.time()
    CPTAClist = image_ids_in(CPTACpath, 'CPTAC')
    TCGAlist = image_ids_in(TCGApath, 'TCGA')

    for i in CPTAClist:
        matchrow = ref.loc[ref['name'] == i[1]]
        if matchrow.empty:
            continue
        try:
            os.mkdir("../tiles/{}".format(i[1]))
            dup = False
        except(FileExistsError):
            dup = True
            pass
        for m in range(3):
            if m == 0:
                tff = 1
                level = 0
            elif m == 1:
                tff = 2
                level = 0
            else:
                tff = 1
                level = 1
            otdir = "../tiles/{}/level{}".format(i[1], str(m))
            try:
                os.mkdir(otdir)
            except(FileExistsError):
                pass
            try:
                n_x, n_y, raw_img, resx, resy, ct = Slicer.tile(image_file='CPTAC/'+i[0], outdir=otdir,
                                                                level=level, dp=dup, ft=tff)
            except(IndexError):
                pass
            if len(os.listdir(otdir)) < 2:
                shutil.rmtree(otdir, ignore_errors=True)

    for i in TCGAlist:
        matchrow = ref.loc[ref['name'] == i[1]]
        if matchrow.empty:
            continue
        try:
            os.mkdir("../tiles/{}".format(i[1]))
            dup = False
        except(FileExistsError):
            dup = True
            pass
        for m in range(3):
            if m == 0:
                tff = 2
                level = 0
            elif m == 1:
                tff = 1
                level = 1
            else:
                tff = 2
                level = 1
            otdir = "../tiles/{}/level{}".format(i[1], str(m))
            try:
                os.mkdir(otdir)
            except(FileExistsError):
                pass
            try:
                n_x, n_y, raw_img, resx, resy, ct = Slicer.tile(image_file='TCGA/'+i[0], outdir=otdir,
                                                                level=level, dp=dup, ft=tff)
            except(IndexError):
                pass
            if len(os.listdir(otdir)) < 2:
                shutil.rmtree(otdir, ignore_errors=True)


    print("--- %s seconds ---" % (time.time() - start_time))

    # # Time measure tool
    # start_time = time.time()
    # print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    if not os.path.isdir('../tiles'):
        os.mkdir('../tiles')
    cut()


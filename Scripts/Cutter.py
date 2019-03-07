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


if __name__ == "__main__":
    CPTACpath = '../images/CPTAC/'
    TCGApath = '../images/TCGA/'
    CPTAC_ref = pd.read_csv('../Filtered_joined_PID.csv', header=0)
    TCGA_ref = pd.read_csv('../new_TCGA_list.csv', header=0)

    # Get all images in the root directory
    def image_ids_in(root_dir, mode, ignore=['.DS_Store', 'dict.csv']):
        ids = []
        for id in os.listdir(root_dir):
            if id in ignore:
                print('Skipping ID:', id)
            else:
                if mode == 'CPTAC':
                    dirname = id.split('_')[-2]
                    ids.append((id, dirname))
                if mode == 'TCGA':
                    dirname = id.split('.')[0]
                    ids.append((id, dirname))
        return ids

    # cut tiles with coordinates in the name (exclude white)
    start_time = time.time()
    CPTAClist = image_ids_in(CPTACpath, 'CPTAC')
    TCGAlist = image_ids_in(TCGApath, 'TCGA')
    for DIR in ("../tiles/", "../tiles/POLE/", "../tiles/Serous-like/", "../tiles/MSI/", "../tiles/Endometrioid/"):
        try:
            os.mkdir(DIR)
        except(FileExistsError):
            pass

    for i in CPTAClist:
        matchrow = CPTAC_ref.loc[CPTAC_ref['Parent Sample ID(s)'] == i[1]]
        if matchrow.empty:
            continue
        label = matchrow['Subtype'].to_string(index=False, header=False).strip()
        print(label)
        try:
            os.mkdir("../tiles/{}/{}".format(label, i[1]))
        except(FileExistsError):
            pass
        for level in range(3):
            otdir = "../tiles/{}/{}/level{}".format(label, i[1], str(level))
            try:
                os.mkdir(otdir)
            except(FileExistsError):
                pass
            try:
                n_x, n_y, raw_img, resx, resy, ct = Slicer.tile(image_file='CPTAC/'+i[0], outdir=otdir, level=level)
            except(IndexError):
                pass
            if len(os.listdir(otdir)) < 2:
                shutil.rmtree(otdir, ignore_errors=True)

    for i in TCGAlist:
        matchrow = TCGA_ref.loc[TCGA_ref['File Name'] == i[0]]
        if matchrow.empty:
            continue
        label = matchrow['label'].to_string(index=False, header=False).strip()
        print(label)
        try:
            os.mkdir("../tiles/{}/{}".format(label, i[1]))
        except(FileExistsError):
            pass
        for level in range(3):
            if level != 0:
                otdir = "../tiles/{}/{}/level{}".format(label, i[1], str(level))
                try:
                    os.mkdir(otdir)
                except(FileExistsError):
                    pass
                try:
                    n_x, n_y, raw_img, resx, resy, ct = Slicer.tile(image_file='TCGA/'+i[0], outdir=otdir, level=(level+1))
                except(IndexError):
                    pass
                if len(os.listdir(otdir)) < 2:
                    shutil.rmtree(otdir, ignore_errors=True)

            else:
                if label == 'Endometrioid':
                    rand = np.random.rand(1)[0]
                    if rand < 0.1:
                        otdir = "../tiles/{}/{}/level{}".format(label, i[1], str(level))
                        try:
                            os.mkdir(otdir)
                        except(FileExistsError):
                            pass
                        try:
                            n_x, n_y, raw_img, resx, resy, ct = Slicer.tile(image_file='TCGA/' + i[0], outdir=otdir,
                                                                            level=(level+1))
                        except(IndexError):
                            pass
                        if len(os.listdir(otdir)) < 2:
                            shutil.rmtree(otdir, ignore_errors=True)
                else:
                    otdir = "../tiles/{}/{}/level{}".format(label, i[1], str(level))
                    try:
                        os.mkdir(otdir)
                    except(FileExistsError):
                        pass
                    try:
                        n_x, n_y, raw_img, resx, resy, ct = Slicer.tile(image_file='TCGA/' + i[0], outdir=otdir,
                                                                        level=(level + 1))
                    except(IndexError):
                        pass
                    if len(os.listdir(otdir)) < 2:
                        shutil.rmtree(otdir, ignore_errors=True)


    print("--- %s seconds ---" % (time.time() - start_time))

    # # Time measure tool
    # start_time = time.time()
    # print("--- %s seconds ---" % (time.time() - start_time))






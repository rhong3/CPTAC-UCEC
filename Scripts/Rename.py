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


def rename():
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

    TCGApp = pd.DataFrame(TCGAlist, columns=['id', 'dir', 'sld'])
    TCGAcc = TCGApp['dir'].value_counts()
    TCGAcc = TCGAcc[TCGAcc > 1].index.tolist()


    # CPTAC
    for i in CPTAClist:
        matchrow = ref.loc[ref['name'] == i[1]]
        if matchrow.empty:
            continue
        if i[1] not in CPTACcc:
            for m in range(3):
                for q in os.listdir("../tiles/{}/level{}".format(i[1], str(m))):
                    if '.png' in q:
                        old = q.split('.pn')[0]
                        os.rename("../tiles/{}/level{}/{}".format(i[1], str(m), q),
                                  "../tiles/{}/level{}/{}_{}.png".format(i[1], str(m), old, str(i[2])))
                    elif '.csv' in q:
                        old = q.split('.cs')[0]
                        os.rename("../tiles/{}/level{}/{}".format(i[1], str(m), q),
                                  "../tiles/{}/level{}/{}_{}.csv".format(i[1], str(m), str(i[2]), old))
    # TCGA
    for i in TCGAlist:
        matchrow = ref.loc[ref['name'] == i[1]]
        if matchrow.empty:
            continue
        if i[1] not in TCGAcc:
            for m in range(3):
                for q in os.listdir("../tiles/{}/level{}".format(i[1], str(m))):
                    if '.png' in q:
                        old = q.split('.pn')[0]
                        os.rename("../tiles/{}/level{}/{}".format(i[1], str(m), q),
                                  "../tiles/{}/level{}/{}_{}.png".format(i[1], str(m), old, str(i[2])))
                    elif '.csv' in q:
                        old = q.split('.cs')[0]
                        os.rename("../tiles/{}/level{}/{}".format(i[1], str(m), q),
                                  "../tiles/{}/level{}/{}_{}.csv".format(i[1], str(m), str(i[2]), old))

    print("--- %s seconds ---" % (time.time() - start_time))


# Run as main
if __name__ == "__main__":
    rename()


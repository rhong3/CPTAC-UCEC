#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Make tSNE mosaic

Created on Wed Jul 17 10:16:21 2019

@author: lwk, RH
"""

import pandas as pd
from PIL import Image
import sys
import os

filename = sys.argv[1]   # DEFAULT='tSNE_P_N'
bin = int(sys.argv[2])   # DEFAULT=50
size = int(sys.argv[3])  # DEFAULT=299
pdmd = sys.argv[4]
dirr = sys.argv[5]
outim = sys.argv[6]


# random select representative images and output the file paths
def sample(dat, md, bins):

    if md == 'subtype':
        classes = 4
        redict = {0: 'MSI_score', 1: 'Endometrioid_score', 2: 'Serous-like_score', 3: 'POLE_score'}
    elif md == 'histology':
        redict = {0: 'Endometrioid_score', 1: 'Serous_score'}
        classes = 2
    elif md == 'MSIst':
        redict = {0: 'MSI-H_score', 1: 'MSS_score'}
        classes = 2
    else:
        redict = {0: 'NEG_score', 1: 'POS_score'}
        classes = 2
    sampledls = []
    for m in range(classes):
        for i in range(bins):
            for j in range(bins):
                try:
                    sub = dat.loc[(dat['x_int'] == i) & (dat['y_int'] == j)
                                    & (dat[redict[m]] > 0.8) & (dat['True_label'] == m)]
                    picked = sub.sample(1, replace=False)
                    for idx, row in picked.iterrows():
                        sampledls.append([row['path'], row['x_int'], row['y_int']])
                except ValueError:
                    pass
    samples = pd.DataFrame(sampledls, columns=['impath', 'x_int', 'y_int'])
    return samples


if __name__ == "__main__":
    # dirls = dirr.split(',')

    ### special ###
    dirls = []
    for n in range(6):
        num = str(n+1)
        genes = ['ARID1A', 'ARID5B', 'ATM', 'BRCA2', 'CTCF', 'CTNNB1', 'EGFR', 'ERBB2',
                        'FBXW7', 'FGFR2', 'JAK1', 'KRAS', 'MLH1', 'MTOR', 'PIK3CA', 'PIK3R1', 'PIK3R2', 'PPP2R1A',
                        'PTEN', 'RPL22', 'TP53']
        for g in genes:
            dirls.append('I{}{}'.format(num, g))
    ### special ###

    for i in dirls:
        try:
            ipdat = pd.read_csv('../Results/NL3/{}/out/{}.csv'.format(i, filename))
            imdat = sample(ipdat, pdmd, bin)
            imdat.to_csv('../Results/NL3/{}/out/tsne_selected.csv'.format(i), index=False)

            new_im = Image.new(mode='RGB', size=(size*bin, size*bin), color='white')

            for rows in imdat.itertuples():
                impath = rows.impath
                x = rows.x_int
                y = rows.y_int
                try:
                    im = Image.open(impath)
                    im.thumbnail((size, size))
                    new_im.paste(im, ((x-1)*size, (bin-y)*size))
                except FileNotFoundError:
                    print(impath)
                    pass
            new_im.save(os.path.abspath('../Results/NL3/{}/out/{}.jpeg'.format(i, outim)), "JPEG")
            print('{} done'.format(i))
        except FileNotFoundError:
            print('{} passed'.format(i))
            pass



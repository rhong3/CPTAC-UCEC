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

filename=sys.argv[1]
bin=sys.argv[2]
size=sys.argv[3]
pdmd = sys.argv[4]
dirr = sys.argv[5]
ipdat = pd.read_csv(filename, header=0)


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
                sub = imdat.loc[(dat['x_int'] == i) & (dat['y_int'] == j)
                                & (dat[redict[m]] > 0.8) & (dat['True_label'] == m)]
                picked = sub.sample(1, replace=False)
                for idx, row in picked.iterrows():
                    sampledls.extend([row['path'], row['x_int'], row['y_int']])
    samples = pd.DataFrame(sampledls, columns=['impath', 'x_int', 'y_int'])
    return samples


imdat = sample(ipdat, pdmd, bin)
imdat.to_csv('../Results/{}/out/tsne_selected.csv'.format(dirr), index=False)
new_im = Image.new(mode='RGB', size=(size*bin,size*bin), color='white')

for row in imdat.itertuples():
    impath = row.impath
    x = row.x_int
    y = row.y_int
    im = Image.open(impath)
    im.thumbnail((size, size))
    new_im.paste(im, ((x-1)*size, (bin-y)*size))
    
new_im.save('../Results/{}/out/model_manifold_neg.jpeg'.format(dirr))


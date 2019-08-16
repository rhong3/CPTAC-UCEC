#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Make tSNE mosaic (TO BE MODIFIED)

Created on Wed Jul 17 10:16:21 2019

@author: lwk
"""


import pandas as pd
from PIL import Image

bins=50
size=100

filename='selected_tiles_neg.csv'
imdat=pd.read_csv(filename,header=0)

new_im = Image.new(mode='RGB',size=(size*bins,size*bins),color='white')

for row in imdat.itertuples():
    impath=row.impath
    x=row.x_int
    y=row.y_int
    im = Image.open(impath)
    im.thumbnail((size,size))
    new_im.paste(im,((x-1)*size,(bins-y)*size))
    
new_im.save('LUAD_model_manifold_neg.jpeg')
    

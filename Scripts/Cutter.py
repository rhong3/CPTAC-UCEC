# Tile a real scn file, load a trained model and run the test.
"""
Created on 11/01/2018

@author: RH
"""

import time
import matplotlib
import os
matplotlib.use('Agg')
import Slicer

path = '../images'

def image_ids_in(root_dir, ignore=['.DS_Store']):
    ids = []
    for id in os.listdir(root_dir):
        if id in ignore:
            print('Skipping ID:', id)
        else:
            ids.append(id)
    return ids

# cut tiles with coordinates in the name (exclude white)
start_time = time.time()
svslist = image_ids_in(path)
os.mkdir("../tiles/")
for i in svslist:
    otdir = "../tiles/"+i
    os.mkdir(otdir)
    n_x, n_y, raw_img, resx, resy, imgs, ct = Slicer.tile(image_file = i, outdir = otdir)
print("--- %s seconds ---" % (time.time() - start_time))

# # Time measure tool
# start_time = time.time()
# print("--- %s seconds ---" % (time.time() - start_time))






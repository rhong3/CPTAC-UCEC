"""
Tile svs/scn files

Created on 11/01/2018

@author: RH
"""

import time
import matplotlib
import os
matplotlib.use('Agg')
import Slicer

path = '../images'


# Get all images in the root directory
def image_ids_in(root_dir, ignore=['.DS_Store', 'dict.csv']):
    ids = []
    for id in os.listdir(root_dir):
        if id in ignore:
            print('Skipping ID:', id)
        else:
            dirname = id.split('_')[-2]
            ids.append((id, dirname))
    return ids

# cut tiles with coordinates in the name (exclude white)
start_time = time.time()
svslist = image_ids_in(path)
for DIR in ("../tiles/", "../tiles/level0/", "../tiles/level1/", "../tiles/level2/"):
    try:
        os.mkdir(DIR)
    except(FileExistsError):
        pass

for level in range(3):
    for i in svslist:
        otdir = "../tiles/level{}/{}".format(str(level), i[1])
        try:
            os.mkdir(otdir)
        except(FileExistsError):
            pass
        try:
            n_x, n_y, raw_img, resx, resy, imgs, ct = Slicer.tile(image_file=i[0], outdir=otdir, level=level)
        except(IndexError):
            pass
        if len(os.listdir(otdir)) == 0:
            os.rmdir(otdir)
print("--- %s seconds ---" % (time.time() - start_time))

# # Time measure tool
# start_time = time.time()
# print("--- %s seconds ---" % (time.time() - start_time))






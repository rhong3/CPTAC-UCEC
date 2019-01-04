"""
Tile a real scn/svs file, load a trained model and run the test.

Created on 11/01/2018

@author: RH
"""

import get_tilev3
import time
import matplotlib
matplotlib.use('Agg')
import os
import sys
import numpy as np
import data_input
import cnn3
import pandas as pd
import cv2
import skimage.morphology as mph

dirr = sys.argv[1]  # name of output directory
imgfile = sys.argv[2]  # input scn/svs name
bs = sys.argv[3]  # batch size
md = sys.argv[4]  # loaded model's structure type
modeltoload = sys.argv[5]  # name of trained model to be loaded
metadir = sys.argv[6]  # metagraph directory
bs = int(bs)

IMG_DIM = 299

INPUT_DIM = [bs, IMG_DIM, IMG_DIM, 3]  # default image size

# default hyperparameters
HYPERPARAMS = {
    "batch_size": bs,
    "dropout": 0.8,
    "learning_rate": 1E-4
}

# path of directories
LOG_DIR = "../Neutrophil"
data_dir = "../Neutrophil/{}/data".format(dirr)
out_dir = "../Neutrophil/{}/out".format(dirr)
METAGRAPH_DIR = "../Neutrophil/{}".format(metadir)
file_DIR = "../Neutrophil/{}".format(dirr)

# make directories
try:
    os.mkdir(METAGRAPH_DIR)
except(FileExistsError):
    pass

try:
    os.mkdir(file_DIR)
except(FileExistsError):
    pass

try:
    os.mkdir(data_dir)
except(FileExistsError):
    pass

try:
    os.mkdir(out_dir)
except(FileExistsError):
    pass

try:
    os.mkdir(LOG_DIR)
except(FileExistsError):
    pass


# load image tiles
def loader(images, bs, ct):
    dataset = data_input.DataSet(bs, ct, images=images)
    return dataset


# main function for real test image prediction
def test(images, count, bs, to_reload=None):
    m = cnn3.INCEPTION(INPUT_DIM, HYPERPARAMS, meta_graph=to_reload, log_dir=LOG_DIR, meta_dir=LOG_DIR, model=md)

    print("Loaded! Ready for test!")
    HE = loader(images, bs, count)
    m.inference(HE, dirr, Not_Realtest=False)


start_time = time.time()
# cut tiles with coordinates in the name (exclude white)
n_x, n_y, raw_img, resx, resy, imgs, ct = get_tilev3.tile(image_file = imgfile, outdir = out_dir)
print("--- %s seconds ---" % (time.time() - start_time))
# load tiles dictionary
dict = pd.read_csv(out_dir+'/dict.csv', header=0)
# predictions on tiles
test(imgs, ct, bs, to_reload=modeltoload)
# load dictionary of predictions on tiles
teresult = pd.read_csv(out_dir+'/Test.csv', header=0)
# join 2 dictionaries
joined = pd.merge(dict, teresult, how='inner', on=['Num'])
# save joined dictionary
joined.to_csv(out_dir+'/finaldict.csv', index=False)

# output heat map of pos and neg.
# initialize a graph and for each RGB channel
opt = np.full((n_x, n_y), 0)
hm_R = np.full((n_x, n_y), 0)
hm_G = np.full((n_x, n_y), 0)
hm_B = np.full((n_x, n_y), 0)

print(np.shape(opt))

lbdict = {0: 'MSI', 1: 'Endometroid', 2: 'Serous-like', 3: 'POLE'}
poscsv = joined.loc[joined['Prediction'] == 1]
for index, row in poscsv.iterrows():
    opt[int(row["X_pos"]), int(row["Y_pos"])] = 255
    hm_R[int(row["X_pos"]), int(row["Y_pos"])] = 255
    hm_G[int(row["X_pos"]), int(row["Y_pos"])] = int((1-(row["pos_score"]-0.5)*2)*255)
    hm_B[int(row["X_pos"]), int(row["Y_pos"])] = int((1 - (row["pos_score"] - 0.5) * 2) * 255)

negcsv = joined.loc[joined['Prediction'] == 0]
for index, row in negcsv.iterrows():
    opt[int(row["X_pos"]), int(row["Y_pos"])] = 255
    hm_B[int(row["X_pos"]), int(row["Y_pos"])] = 255
    hm_G[int(row["X_pos"]), int(row["Y_pos"])] = int((1-(row["neg_score"]-0.5)*2)*255)
    hm_R[int(row["X_pos"]), int(row["Y_pos"])] = int((1 - (row["neg_score"] - 0.5) * 2) * 255)

opt = opt.repeat(5, axis=0).repeat(5, axis=1)
opt = mph.remove_small_objects(opt.astype(bool), min_size=500, connectivity=2).astype(np.uint8)

ori_img = cv2.resize(raw_img, (np.shape(opt)[0]+resx, np.shape(opt)[1]+resy))
ori_img = ori_img[:np.shape(opt)[1], :np.shape(opt)[0], :3]
tq = ori_img[:,:,0]
ori_img[:,:,0] = ori_img[:,:,2]
ori_img[:,:,2] = tq
cv2.imwrite(out_dir+'/Original_scaled.png', ori_img)

topt = np.transpose(opt)
opt = np.full((np.shape(topt)[0], np.shape(topt)[1], 3), 0)
opt[:,:,0] = topt
opt[:,:,1] = topt
opt[:,:,2] = topt
cv2.imwrite(out_dir+'/Mask.png', opt*255)

hm_R = np.transpose(hm_R)
hm_G = np.transpose(hm_G)
hm_B = np.transpose(hm_B)
hm_R = hm_R.repeat(5, axis=0).repeat(5, axis=1)
hm_G = hm_G.repeat(5, axis=0).repeat(5, axis=1)
hm_B = hm_B.repeat(5, axis=0).repeat(5, axis=1)
hm = np.dstack([hm_B, hm_G, hm_R])
hm = hm*opt
cv2.imwrite(out_dir+'/HM.png', hm)

ori_img = ori_img*opt
overlay = ori_img * 0.65 + hm * 0.35
cv2.imwrite(out_dir+'/Overlay.png', overlay)


# # Time measure tool
# start_time = time.time()
# print("--- %s seconds ---" % (time.time() - start_time))






import pandas as pd
from shutil import copy2
import os

TODO = ['X1his', 'F1FAT1', 'F4MSIst', 'X1PTEN', 'X1TP53', 'X2SL', 'X2ZFHX3', 'X3CNVH']
try:
    os.mkdir('../Results/NL5/FPFN')
except FileExistsError:
    pass

for i in TODO:
    if i == 'X1his':
        POS_score = 'Serous_score'
    elif i == 'F4MSIst':
        POS_score = 'MSI-H_score'
    else:
        POS_score = 'POS_score'
    try:
        os.mkdir('../Results/NL5/FPFN/{}'.format(i))
    except FileExistsError:
        pass
    try:
        os.mkdir('../Results/NL5/FPFN/{}/FP'.format(i))
    except FileExistsError:
        pass
    try:
        os.mkdir('../Results/NL5/FPFN/{}/FN'.format(i))
    except FileExistsError:
        pass
    ref = pd.read_csv('../Results/NL5/{}/out/Test_tile.csv'.format(i), header=0)
    FP = ref[(ref[POS_score] > 0.95) & (ref['label'] == 0)]
    FN = ref[(ref[POS_score] < 0.05) & (ref['label'] == 1)]
    for idx, row in FP.iterrows():
        copy2(row['L0path'], '../Results/NL5/FPFN/{}/FP/'.format(i))
    for idx, row in FN.iterrows():
        copy2(row['L0path'], '../Results/NL5/FPFN/{}/FN/'.format(i))

print('Done!')

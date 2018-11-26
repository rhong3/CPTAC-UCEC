import os
import pandas as pd
import sklearn.utils as sku
import numpy as np

tile_path = "../tiles/"

def image_ids_in(root_dir, ignore=['.DS_Store']):
    ids = []
    for id in os.listdir(root_dir):
        if id in ignore:
            print('Skipping ID:', id)
        else:
            ids.append(id)
    return ids


def inlist_check(pdlist, dir):
    isin = []
    Not = []
    tiles = image_ids_in(dir)
    pd = pdlist["Parent Sample ID(s)"].tolist()
    for a in tiles:
        if a in pd:
            isin.append(a)
        else:
            Not.append(a)
    return isin, Not


def samplesum(pdlist, path):
    data = []
    lbdict ={'MSI': 1, 'Endometroid': 2, 'Serous-like' : 3, 'POLE' : 4}
    for idx, row in pdlist.iterrows():
        folder = row["Parent Sample ID(s)"]
        label = row["Subtype"]
        try:
            imgs = image_ids_in(path+folder)
            for i in imgs:
                imgsdir = path+folder+'/'+i
                data.append([imgsdir, int(lbdict[label])])
        except:
            print(folder+' not in tiles!')
    datapd = pd.DataFrame(data, columns = ['path', 'label'])
    datapd = sku.shuffle(datapd)

    return datapd


pan = pd.read_csv('../new_joined_PID.csv', header = 0)
pl, nl = inlist_check(pan, '../tiles')
with open('../inlist.csv','w') as f:
    f.write(','.join(pl))
    f.close()
with open('../Notinlist.csv','w') as f:
    f.write(','.join(nl))
    f.close()
all = samplesum(pan, '../tiles/')
all.to_csv('../tiles/all.csv', header = True, index = False)

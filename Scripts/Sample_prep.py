import os
import pandas as pd
import sklearn.utils as sku
import numpy as np

tile_path = "../tiles/"

def image_ids_in(root_dir, ignore=['.DS_Store','dict.csv','all.csv', 'tr_sample.csv', 'te_sample.csv']):
    ids = []
    for id in os.listdir(root_dir):
        if id in ignore:
            print('Skipping ID:', id)
        else:
            ids.append(id)
    return ids


def inlist_check(pdlist = pd.read_csv('../new_joined_PID.csv', header = 0), dir = "../tiles/"):
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


def samplesum(pdlist = pd.read_csv('../new_joined_PID.csv', header = 0), path = "../tiles/"):
    data = []
    lbdict ={'MSI': 0, 'Endometroid': 1, 'Serous-like' : 2, 'POLE' : 3}
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


def set_sep(all, path = "../tiles/", cut=0.1):
    test, train = np.split(all, [int(cut*len(all))])
    test.to_csv(path+'te_sample.csv', header=True, index=False)
    train.to_csv(path + 'tr_sample.csv', header=True, index=False)

    return train, test


if __name__ == "__main__":
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

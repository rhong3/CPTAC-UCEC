import cv2
import pandas as pd
import numpy as np
import os


def tile_ids_in(slide, level, root_dir, label, ignore=['.DS_Store','dict.csv', 'all.csv']):
    ids = []
    try:
        for id in os.listdir(root_dir):
            if id in ignore:
                print('Skipping ID:', id)
            else:
                ids.append([slide, level, root_dir+'/'+id, label])
    except FileNotFoundError:
        print('Ignore:', root_dir)

    return ids


# read images
def load_image(addr):
    img = cv2.imread(addr)
    img = img.astype(np.float32)
    return img


# loading images for dictionaries and generate tfrecords
def loader(totlist_dir):
    RGB=[]
    all = pd.read_csv(totlist_dir+'/All_images.csv', header=0)
    tiles_list = []
    for idx, row in all.iterrows():
        tile_ids = tile_ids_in(row['slide'], row['level'], row['path'], row['label'])
        tiles_list.extend(tile_ids)
    tiles = pd.DataFrame(tiles_list, columns=['slide', 'level', 'path', 'label'])

    imlist = tiles['path'].values.tolist()

    for i in range(len(imlist)):
        try:
            # Load the image
            img = load_image(imlist[i])
            the_imagea = np.array(img)[:, :, :3]
            the_imagea = np.nan_to_num(the_imagea)
            mask = (the_imagea[:, :, :3] > 200).astype(np.uint8)
            maskb = (the_imagea[:, :, :3] < 50).astype(np.uint8)
            mask = (~(mask[:, :, 0] * mask[:, :, 1] * mask[:, :, 2]).astype(bool)).astype(np.uint8)
            maskb = (~(maskb[:, :, 0] * maskb[:, :, 1] * maskb[:, :, 2]).astype(bool)).astype(np.uint8)
            mask = mask*maskb
            masksum = np.sum(mask)
            BB = np.sum(the_imagea[:, :, 0]*mask)/masksum
            B = (the_imagea[:, :, 0]*mask).ravel()[np.flatnonzero(the_imagea[:, :, 0]*mask)]
            GG = np.sum(the_imagea[:, :, 1]*mask)/masksum
            G = (the_imagea[:, :, 1]*mask).ravel()[np.flatnonzero(the_imagea[:, :, 1]*mask)]
            RR = np.sum(the_imagea[:, :, 2]*mask)/masksum
            R = (the_imagea[:, :, 2]*mask).ravel()[np.flatnonzero(the_imagea[:, :, 2]*mask)]
            RGB.append([imlist[i], RR, np.percentile(R, 25), np.percentile(R, 75), np.std(R),
                        GG, np.percentile(G, 25), np.percentile(G, 75), np.std(G),
                        BB, np.percentile(B, 25), np.percentile(B, 75), np.std(B)])
        except AttributeError:
            print('Error image:'+imlist[i])
            pass

    RGBdf = pd.DataFrame(RGB, columns=['Img', 'Redmean', 'Red25', 'Red75', 'Redstd',
                                       'Greenmean', 'Green25', 'Green75', 'Greenstd',
                                       'Bluemean', 'Blue25', 'Blue75', 'Bluestd'])

    RGBdf.to_csv('../Results/RGB.csv', index=False, header=True)

    Rmean = RGBdf['Redmean'].mean()
    Gmean = RGBdf['Greenmean'].mean()
    Bmean = RGBdf['Bluemean'].mean()
    Rstd = RGBdf['Redstd'].mean()
    Gstd = RGBdf['Greenstd'].mean()
    Bstd = RGBdf['Bluestd'].mean()
    Raqt = RGBdf['Red25'].mean()
    Gaqt = RGBdf['Green25'].mean()
    Baqt = RGBdf['Blue25'].mean()
    Rbqt = RGBdf['Red75'].mean()
    Gbqt = RGBdf['Green75'].mean()
    Bbqt = RGBdf['Blue75'].mean()

    print("Red Mean of: mean={}; std={}; 25pct={}; 75pct={}".format(str(Rmean), str(Rstd), str(Raqt), str(Rbqt)))
    print("Green Mean of: mean={}; std={}; 25pct={}; 75pct={}".format(str(Gmean), str(Gstd), str(Gaqt), str(Gbqt)))
    print("Blue Mean of: mean={}; std={}; 25pct={}; 75pct={}".format(str(Bmean), str(Bstd), str(Baqt), str(Bbqt)))


if __name__ == "__main__":
    loader('../tiles')
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
            GG = np.sum(the_imagea[:, :, 1]*mask)/masksum
            RR = np.sum(the_imagea[:, :, 2]*mask)/masksum
            RGB.append([imlist[i], RR, GG, BB])
        except AttributeError:
            print('Error image:'+imlist[i])
            pass

    RGBdf = pd.DataFrame(RGB, columns=['Img', 'Red', 'Green', 'Blue'])

    RGBdf.to_csv('../Results/RGB.csv', index=False, header=True)

    Rmean = RGBdf['Red'].mean()
    Gmean = RGBdf['Green'].mean()
    Bmean = RGBdf['Blue'].mean()
    Rstd = RGBdf['Red'].std()
    Gstd = RGBdf['Green'].std()
    Bstd = RGBdf['Blue'].std()

    print("Red mean={}; std={}".format(str(Rmean), str(Rstd)))
    print("Green mean={}; std={}".format(str(Gmean), str(Gstd)))
    print("Blue mean={}; std={}".format(str(Bmean), str(Bstd)))


if __name__ == "__main__":
    loader('../tiles')
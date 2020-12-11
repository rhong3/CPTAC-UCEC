"""
Tile svs/scn files for NYU samples

Created on 10/21/2020

@author: RH
"""
import time
import matplotlib
import os
import shutil
import pandas as pd
matplotlib.use('Agg')
import Slicer_NYU as Slicer


# cut; each level is 2 times difference (10x, 5x, 2.5x)
def cut():
    # load standard image for normalization
    ref = pd.read_csv('../NYU/IHC_sum.csv', header=0, usecols=['Patient_ID', 'Slide_ID', 'file'])
    # cut tiles with coordinates in the name (exclude white)
    start_time = time.time()

    # CPTAC
    for idx, row in ref.iterrows():
        try:
            os.mkdir("../tiles/{}".format(row['Patient_ID']))
        except FileExistsError:
            pass
        for m in range(1, 4):
            if m == 0:
                tff = 1
                level = 0
            elif m == 1:
                tff = 2
                level = 0
            elif m == 2:
                tff = 1
                level = 1
            elif m == 3:
                tff = 2
                level = 1
            otdir = "../tiles/{}/level{}".format(row['Patient_ID'], str(m))
            try:
                os.mkdir(otdir)
            except FileExistsError:
                pass
            try:
                n_x, n_y, raw_img, ct = Slicer.tile(image_file='NYU/'+row['file'], outdir=otdir,
                                                                level=level, std_img=std, dp=row['Slide_ID'], ft=tff)
            except Exception as e:
                print(e)
                pass
            if len(os.listdir(otdir)) < 2:
                shutil.rmtree(otdir, ignore_errors=True)

    print("--- %s seconds ---" % (time.time() - start_time))
    # # Time measure tool
    # start_time = time.time()
    # print("--- %s seconds ---" % (time.time() - start_time))


# Run as main
if __name__ == "__main__":
    if not os.path.isdir('../tiles'):
        os.mkdir('../tiles')
    cut()


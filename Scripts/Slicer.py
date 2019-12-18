"""
Tile real scn/svs files; used by Cutter.py

Created on 11/19/2018

*** Removed imlist storage to minimize memory usage 01/24/2019 ***

@author: RH
"""
from openslide import OpenSlide
import numpy as np
import pandas as pd
import multiprocessing as mp
import staintools
from PIL import Image


# check if a tile is background or not; return a blank pixel percentage score
def bgcheck(img, ts):
    the_imagea = np.array(img)[:, :, :3]
    the_imagea = np.nan_to_num(the_imagea)
    mask = (the_imagea[:, :, :3] > 200).astype(np.uint8)
    maskb = (the_imagea[:, :, :3] < 50).astype(np.uint8)
    # greya = ((np.ptp(the_imagea[0])) < 100).astype(np.uint8)
    # greyb = ((np.ptp(the_imagea[1])) < 100).astype(np.uint8)
    # greyc = ((np.ptp(the_imagea[2])) < 100).astype(np.uint8)
    # grey = greya * greyb * greyc
    mask = mask[:, :, 0] * mask[:, :, 1] * mask[:, :, 2]
    maskb = maskb[:, :, 0] * maskb[:, :, 1] * maskb[:, :, 2]
    white = (np.sum(mask) + np.sum(maskb)) / (ts * ts)
    return white


# Tile color normalization
def normalization(img, sttd):
    img = np.array(img)[:, :, :3]
    img = staintools.LuminosityStandardizer.standardize(img)
    normalizer = staintools.StainNormalizer(method='vahadane')
    normalizer.fit(sttd)
    img = normalizer.transform(img)
    img = Image.fromarray(img.astype('uint8'), 'RGB')
    return img


# tile method; slp is the scn/svs image; n_y is the number of tiles can be cut on y column to be cut;
# x and y are the upper left position of each tile; tile_size is tile size; stepsize of each step; x0 is the row to cut.
# outdir is the output directory for images;
# imloc record each tile's relative and absolute coordinates; imlist is a list of cut tiles (Removed 01/24/2019).
def v_slide(slp, n_y, x, y, tile_size, stepsize, x0, outdir, level, dp, std):
    # pid = os.getpid()
    # print('{}: start working'.format(pid))
    slide = OpenSlide(slp)
    imloc = []
    y0 = 0
    target_x = x0 * stepsize
    image_x = (target_x + x)*(4**level)
    while y0 < n_y:
        target_y = y0 * stepsize
        image_y = (target_y + y)*(4**level)
        img = slide.read_region((image_x, image_y), level, (tile_size, tile_size))
        wscore = bgcheck(img, tile_size)
        if wscore < 0.8:
            img = img.resize((299, 299))
            img = normalization(img, std)
            if dp:
                img.save(outdir + "/region_x-{}-y-{}_{}.png".format(image_x, image_y, str(dp)))
                strr = outdir + "/region_x-{}-y-{}_{}.png".format(image_x, image_y, str(dp))
            else:
                img.save(outdir + "/region_x-{}-y-{}.png".format(image_x, image_y))
                strr = outdir + "/region_x-{}-y-{}.png".format(image_x, image_y)
            imloc.append([x0, y0, image_x, image_y, strr])
        y0 += 1
    slide.close()
    return imloc


# image_file is the scn/svs name; outdir is the output directory; path_to_slide is where the scn/svs stored.
# First open the slide, determine how many tiles can be cut, record the residue edges width,
# and calculate the final output prediction heat map size should be. Then, using multithread to cut tiles, and stack up
# tiles and their position dictionaries.
def tile(image_file, outdir, level, std_img, path_to_slide="../images/", dp=None, ft=1):
    slide = OpenSlide(path_to_slide+image_file)
    slp = str(path_to_slide+image_file)
    print(slp)
    print(slide.level_dimensions)

    bounds_width = slide.level_dimensions[level][0]
    bounds_height = slide.level_dimensions[level][1]
    x = 0
    y = 0
    half_width_region = 49*ft
    full_width_region = 299*ft
    stepsize = (full_width_region - half_width_region)

    n_x = int((bounds_width - 1) / stepsize)
    n_y = int((bounds_height - 1) / stepsize)

    lowres = slide.read_region((x, y), level+1, (int(n_x*stepsize/4), int(n_y*stepsize/4)))
    lowres = np.array(lowres)[:,:,:3]

    x0 = 0
    # create multiporcessing pool
    print(mp.cpu_count())
    pool = mp.Pool(processes=mp.cpu_count())
    tasks = []
    while x0 < n_x:
        task = tuple((slp, n_y, x, y, full_width_region, stepsize, x0, outdir, level, dp, std_img))
        tasks.append(task)
        x0 += 1
    # slice images with multiprocessing
    temp = pool.starmap(v_slide, tasks)
    tempdict = list(temp)
    temp = None
    pool.close()
    pool.join()

    tempdict = list(filter(None, tempdict))
    imloc = []
    list(map(imloc.extend, tempdict))
    imlocpd = pd.DataFrame(imloc, columns = ["X_pos", "Y_pos", "X", "Y", "Loc"])
    imlocpd = imlocpd.sort_values(["X_pos", "Y_pos"], ascending=[True, True])
    imlocpd = imlocpd.reset_index(drop=True)
    imlocpd = imlocpd.reset_index(drop=False)
    imlocpd.columns = ["Num", "X_pos", "Y_pos", "X", "Y", "Loc"]
    if dp:
        imlocpd.to_csv(outdir + "/{}_dict.csv".format(dp), index=False)
    else:
        imlocpd.to_csv(outdir + "/dict.csv", index=False)
    tempdict = None
    ct = len(imloc)
    print(ct)

    return n_x, n_y, lowres, ct



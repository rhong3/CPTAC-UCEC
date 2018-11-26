from openslide import OpenSlide
import numpy as np
import pandas as pd
import multiprocessing as mp


def bgcheck(img):
    the_imagea = np.array(img)[:, :, :3]
    the_imagea = np.nan_to_num(the_imagea)
    mask = (the_imagea[:, :, :3] > 200).astype(np.uint8)
    maskb = (the_imagea[:, :, :3] < 5).astype(np.uint8)
    mask = mask[:, :, 0] * mask[:, :, 1] * mask[:, :, 2]
    maskb = maskb[:, :, 0] * maskb[:, :, 1] * maskb[:, :, 2]
    white = (np.sum(mask) + np.sum(maskb)) / (299 * 299)
    return white


def v_slide(slp, n_y, x, y, tile_size, stepsize, x0):
    # pid = os.getpid()
    # print('{}: start working'.format(pid))
    slide = OpenSlide(slp)
    imloc = []
    imlist = []
    y0 = 0
    target_x = x0 * stepsize
    image_x = target_x + x
    while y0 < n_y:
        target_y = y0 * stepsize
        image_y = target_y + y
        img = slide.read_region((image_x, image_y), 0, (tile_size, tile_size))
        wscore = bgcheck(img)
        if wscore < 0.5:
            imloc.append([x0, y0, target_x, target_y])
            imlist.append(np.array(img)[:, :, :3])
        y0 += 1
    slide.close()
    return imloc, imlist


def tile(image_file, outdir, path_to_slide = "../Neutrophil/"):
    slide = OpenSlide(path_to_slide+image_file)
    slp = str(path_to_slide+image_file)

    assert 'openslide.bounds-height' in slide.properties
    assert 'openslide.bounds-width' in slide.properties
    assert 'openslide.bounds-x' in slide.properties
    assert 'openslide.bounds-y' in slide.properties

    x = int(slide.properties['openslide.bounds-x'])
    y = int(slide.properties['openslide.bounds-y'])
    bounds_height = int(slide.properties['openslide.bounds-height'])
    bounds_width = int(slide.properties['openslide.bounds-width'])

    half_width_region = 49
    full_width_region = 299
    stepsize = full_width_region - half_width_region

    n_x = int((bounds_width - 1) / stepsize)
    n_y = int((bounds_height - 1) / stepsize)

    residue_x = int((bounds_width - n_x * stepsize)/50)
    residue_y = int((bounds_height - n_y * stepsize)/50)
    lowres = slide.read_region((x, y), 2, (int(n_x*stepsize/16), int(n_y*stepsize/16)))
    lowres = np.array(lowres)[:,:,:3]

    x0 = 0
    # create multiporcessing pool
    print(mp.cpu_count())
    pool = mp.Pool(processes=8)
    tasks = []
    while x0 < n_x:
        task = tuple((slp, n_y, x, y, full_width_region, stepsize, x0))
        tasks.append(task)
        x0 += 1
    # slice images with multiprocessing
    temp = pool.starmap(v_slide, tasks)
    tempdict = list(zip(*temp))[0]
    tempimglist = list(zip(*temp))[1]
    temp = None
    pool.close()
    pool.join()

    tempdict = list(filter(None, tempdict))
    imloc = []
    list(map(imloc.extend, tempdict))
    imlocpd = pd.DataFrame(imloc, columns = ["X_pos", "Y_pos", "X", "Y"])
    imlocpd = imlocpd.sort_values(["X_pos", "Y_pos"], ascending=[True, True])
    imlocpd = imlocpd.reset_index(drop=True)
    imlocpd = imlocpd.reset_index(drop=False)
    imlocpd.columns = ["Num", "X_pos", "Y_pos", "X", "Y"]
    imlocpd.to_csv(outdir + "/dict.csv", index = False)
    tempdict = None

    tempimglist = list(filter(None, tempimglist))
    imglist = []
    list(map(imglist.extend, tempimglist))
    ct = len(imglist)
    tempimglist = None
    imglist = np.asarray(imglist)


    return n_x, n_y, lowres, residue_x, residue_y, imglist, ct



'''
Filename: slice_svs.py
Python Version: 3.6.5
Project: Imageomics
Author: Yang Liu
Created date: Oct 17, 2018 2:53 PM
-----
Last Modified: Oct 18, 2018 9:36 AM
Modified By: Yang Liu
-----
License: MIT
http://www.opensource.org/licenses/MIT
'''

import os
import logging
import multiprocessing as mp

from utils import timer
from utils import Paths

import numpy as np
from openslide import OpenSlide
import tables as tb


def v_slide(svs_path, bound_y, queue, tile_size, start_point):
    """
    """
    logger = logging.getLogger(__name__)
    AVG_THRESHOLD = 170
    LAYER = 0
    GREEN_CHANNEL_INDEX = 1
    try:
        svs_file = OpenSlide(svs_path)
        filename = os.path.basename(svs_path)

        x0 = start_point[0]
        y0 = start_point[1]

        pid = os.getpid()
        logger.debug(f'{pid}: start working...')
        logger.debug(f'worker {pid} working on {filename}...')
        while y0 < bound_y:
            img = svs_file.read_region((x0, y0), LAYER, (tile_size, tile_size))
            green_c_avg = np.average(np.array(img)[:, :, GREEN_CHANNEL_INDEX])
            if green_c_avg < AVG_THRESHOLD:
                img = np.array(img)
                img = img[:, :, 0:3]
                queue.put(img)

            y0 += tile_size

        logger.debug(f'{pid}: work finished.')
    finally:
        svs_file.close()


def listener(q, output_path, tile_size):
    """
    """
    try:
        logger = logging.getLogger(__name__)
        filename = os.path.basename(output_path)
        counter = 0
        pid = os.getpid()
        logger.debug(f'Listener running on {pid}...')
        hdf5_file = tb.open_file(output_path, mode='w')
        img_storage = hdf5_file.create_earray(
            hdf5_file.root,
            'training',
            tb.UInt8Atom(),
            shape=(0, tile_size, tile_size, 3)
        )
        while 1:
            counter += 1
            if counter % 100 == 0:
                logger.info(f'{counter} tiles saved in {filename}...')
            try:
                img = q.get()
            except EOFError:
                continue
            if str(img) == 'kill':
                logger.info('Listner closed.')
                hdf5_file.close()
                return None

            img_storage.append(img[None])

    finally:
        hdf5_file.close()


@timer
def slice_svs(svs_path, output_path=None, tile_size=299):
    """
    Slice the svs files into tiles. Tiles sizes are (299, 299).
    Tiles have a half of the tile width overlapping with the tiles beside them
    to ensure all regions are well covered without neutrophil locating on the
    side of the tiles to be ignored in the later prediction.
    The function use multiprocessing method to increase slicing speed.
    Tiles are saved into hdf5 file after sliding.

    input:
        svs_path - path to the directory containing svs files
        output_path - equals to svs_path if not specified
        save_tiles - save tiles images or not

    output:
        hdf5 file saved in neutrophil/data/test folder.
        Tiles will be saved in the same folder if save_tiles flag is set to
        True.
    """
    logger = logging.getLogger(__name__)
    files = os.scandir(svs_path)
    if output_path is None:
        output_path = svs_path
    else:
        try:
            os.makedirs(output_path)
        except os.error:
            pass
        except Exception as e:
            raise Exception('Fail to make output directory.', e)

    for f in files:
        if f.name.endswith('.svs'):
            svs_file = OpenSlide(f.path)

        # get attributes of the scn_file
        x0 = 0
        y0 = 0
        bound_x = int(svs_file.properties['openslide.level[0].width'])
        bound_y = int(svs_file.properties['openslide.level[0].height'])
        logger.debug(f'Filename: {f.name}')
        logger.debug(
            f'x0: {x0}, y0: {y0}, width: {bound_x}, height: {bound_y}')

        start_point = [x0, y0]
        svs_file.close()

        # create multiporcessing pool
        pool = mp.Pool(mp.cpu_count())
        manager = mp.Manager()
        q = manager.Queue()

        tasks = []
        task = [f.path, bound_y, q, tile_size]
        while start_point[0] < bound_x:
            tasks.append(tuple((*task, [start_point[0], start_point[1]])))
            start_point[0] += tile_size

        # run the listener
        watcher = mp.Process(
            target=listener,
            args=(
                q,
                os.path.join(output_path, f.name.split('.')[0] + '.hdf5'),
                tile_size
            )
        )
        watcher.start()

        # slice images with multiprocessing
        pool.starmap(v_slide, tasks)
        pool.close()
        pool.join()
        logger.debug('pool closed.')

        # kill listener
        q.put("kill")
        logger.debug("killer sent.")
        watcher.join()
        logger.debug('listener joined.')

    logging.debug("\nDone!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    slice_svs(Paths.images, os.path.join(Paths.data, 'hdf5'))

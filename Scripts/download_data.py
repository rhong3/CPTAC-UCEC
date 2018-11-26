import os
import socket
import urllib.request
import logging
import multiprocessing
import argparse
import sys


socket.setdefaulttimeout(0.2)


def download_url_in_line(line, save_path):
    image_class = line.split('\t')[0].split('_')[0]
    image_name = line.split('\t')[0] + '.jpg'
    save_dir = os.path.join(save_path, image_class)
    save_name = os.path.join(save_dir, image_name)

    try:
        os.makedirs(save_dir)
    except os.error:
        pass
    image_url = line.split('\t')[-1].strip()
    if image_url.endswith('.jpg'):
        try:
            urllib.request.urlretrieve(image_url, save_name)
            logging.info('{} downloaded...'.format(save_name))
            return
        except socket.gaierror:
            logging.info('{} not available...'.format(image_url))
            return
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            ConnectionResetError,
            socket.timeout,
            Exception
        ):
            logging.info('{} not available...'.format(image_url))
            return
    else:
        logging.info('{} does not link to a jpg file...'.format(image_url))
        return


def main(url_file, max=None):
    logging.basicConfig(level=logging.INFO)
    f = open(url_file, 'r')
    line = f.readline()
    save_path = os.path.join('.', 'data', url_file.split('.')[0][0:-5])
    lines = []
    cnt = 0
    while line:
        try:
            lines.append(tuple((line, save_path)))
            if max is not None:
                cnt += 1
                if cnt == max:
                    break
            line = f.readline()
        except Exception:
            line = f.readline()

    pool = multiprocessing.Pool(os.cpu_count())
    try:
        pool.starmap(download_url_in_line, lines)
        pool.close()
        pool.join()
        logging.info('Download finished.')
    finally:
        pool.close()
        pool.join()
        f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Download image files collected by imageNet database...'
    )
    parser.add_argument(
        '-u', '--url-file', default='imagenet_fall11_urls.txt',
        choices=[
            'imagenet_fall11_urls.txt',
            'imagenet_fall09_urls.txt',
            'imagenet_spring10_urls.txt',
            'imagenet_winter11_urls.txt'
        ]
    )
    parser.add_argument('-m', '--max', type=int)
    args = vars(parser.parse_args(sys.argv[1:]))
    main(**args)

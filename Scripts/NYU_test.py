"""
Test for NYU samples

Created on 10/28/2020

@author: RH
"""
import os
import argparse
import tensorflow as tf
import pandas as pd
import cnn5
import data_input_fusion
import Sample_prep2
import cnn4
import data_input2
import Sample_prep
import NYU_loaders
import matplotlib
matplotlib.use('Agg')

parser = argparse.ArgumentParser()
parser.add_argument('--dirr', type=str, default='trial', help='output directory')
parser.add_argument('--pdmd', type=str, default='histology', help='feature to predict')
parser.add_argument('--mode', type=str, default='X1', help='train or test')
parser.add_argument('--modeltoload', type=str, default='', help='reload trained model')
parser.add_argument('--metadirr', type=str, default='', help='reload trained model in dirr')
parser.add_argument('--reference', type=str, default='../NYU/label.csv', help='reference label file')
opt = parser.parse_args()

print(opt)

if opt.mode in ['F1', 'F2', 'F3', 'F4']:
    sup = True
else:
    sup = False

ref = pd.read_csv(opt.reference, header=0)

# paths to directories
img_dir = '../tiles/'
LOG_DIR = "../Results/{}".format(opt.dirr)
METAGRAPH_DIR = "../Results/{}".format(opt.metadirr)
data_dir = "../Results/{}/data".format(opt.dirr)
out_dir = "../Results/{}/out".format(opt.dirr)

if __name__ == "__main__":
    tf.reset_default_graph()
    # make directories if not exist
    for DIR in (LOG_DIR, METAGRAPH_DIR, data_dir, out_dir):
        try:
            os.mkdir(DIR)
        except FileExistsError:
            pass
    if opt.mode in ['X1', 'X2', 'X3', 'X4', 'F1', 'F2', 'F3', 'F4']:
        bs = 24
        # input image dimension
        INPUT_DIM = [bs, 299, 299, 3]
        # hyper parameters
        HYPERPARAMS = {
            "batch_size": bs,
            "dropout": 0.3,
            "learning_rate": 1E-4,
            "classes": 2,
            "sup": sup
        }
        try:
            tes = pd.read_csv(data_dir + '/te_sample.csv', header=0)
        except FileNotFoundError:
            big_images = []
            if opt.pdmd == 'histology':
                ref = ref.loc[ref['histology_Mixed'] == 0]
                for idx, row in ref.iterrows():
                    if row['histology_Endometrioid'] == 1:
                        big_images.append([row['name'], 0, img_dir + "{}/".format(str(row['name'])), row['age'], row['BMI']])
                    if row['histology_Serous'] == 1:
                        big_images.append([row['name'], 1, img_dir + "{}/".format(str(row['name'])), row['age'], row['BMI']])
            elif opt.pdmd in ['Endometrioid', 'MSI', 'Serous-like', 'POLE']:
                # # special version
                # ref = ref.loc[ref['histology_Mixed'] == 0]
                # ref = ref.loc[ref['histology_Endometrioid'] == 1]
                # # special version

                ref = ref.loc[ref['subtype_0NA'] == 0]
                for idx, row in ref.iterrows():
                    big_images.append(
                        [row['name'], int(row['subtype_{}'.format(opt.pdmd)]), img_dir + "{}/".format(str(row['name'])),
                         row['age'], row['BMI']])
            else:
                ref = ref.dropna(subset=[opt.pdmd])
                for idx, row in ref.iterrows():
                    big_images.append(
                        [row['name'], int(row[opt.pdmd]), img_dir + "{}/".format(str(row['name'])), row['age'], row['BMI']])

            datapd = pd.DataFrame(big_images, columns=['slide', 'label', 'path', 'age', 'BMI'])
            datapd = datapd.dropna()

            test_tiles = pd.DataFrame(columns=['slide', 'label', 'L0path', 'L1path', 'L2path', 'age', 'BMI'])
            for idx, row in datapd.iterrows():
                tile_ids = Sample_prep2.paired_tile_ids_in(row['slide'], row['label'], row['path'], row['age'], row['BMI'])
                test_tiles = pd.concat([test_tiles, tile_ids])
            test_tiles.to_csv(data_dir + '/te_sample.csv', header=True, index=False)
            tes = test_tiles
        tecc = len(tes['label'])
        if not os.path.isfile(data_dir + '/test.tfrecords'):
            NYU_loaders.loaderX(data_dir, 'test')
        m = cnn5.INCEPTION(INPUT_DIM, HYPERPARAMS, meta_graph=opt.modeltoload, log_dir=LOG_DIR,
                           meta_dir=METAGRAPH_DIR, model=opt.mode)
        print("Loaded! Ready for test!")
        if tecc >= bs:
            datasets = data_input_fusion.DataSet(bs, tecc, ep=1, cls=2, mode='test', filename=data_dir + '/test.tfrecords')
            m.inference(datasets, opt.dirr, testset=tes, pmd=opt.pdmd)
        else:
            print("Not enough testing images!")

    else:
        bs = 64
        # input image dimension
        INPUT_DIM = [bs, 299, 299, 3]
        # hyper parameters
        HYPERPARAMS = {
            "batch_size": bs,
            "dropout": 0.3,
            "learning_rate": 1E-4,
            "classes": 2
        }

        try:
            tes = pd.read_csv(data_dir + '/te_sample.csv', header=0)
        except FileNotFoundError:
            allimg = Sample_prep.image_ids_in(img_dir)
            level = 1
            big_images = []
            if opt.pdmd == 'histology':
                allimg = Sample_prep.intersection(ref.loc[ref['histology_Mixed'] == 0]['name'].tolist(), allimg)
                EMimg = Sample_prep.intersection(ref.loc[ref['histology_Endometrioid'] == 1]['name'].tolist(), allimg)
                Serousimg = Sample_prep.intersection(ref.loc[ref['histology_Serous'] == 1]['name'].tolist(), allimg)
                for i in EMimg:
                    big_images.append([i, level, img_dir + "{}/level{}".format(i, level), 0])
                for i in Serousimg:
                    big_images.append([i, level, img_dir + "{}/level{}".format(i, level), 1])
            elif opt.pdmd in ['Endometrioid', 'MSI', 'Serous-like', 'POLE']:
                ref = ref.loc[ref['subtype_0NA'] == 0]

                # ## special version
                # ref = ref.loc[ref['histology_Mixed'] == 0]
                # ref = ref.loc[ref['histology_Endometrioid'] == 1]
                # ## special version

                negimg = Sample_prep.intersection(ref.loc[ref['subtype_{}'.format(opt.pdmd)] == 0]['name'].tolist(), allimg)
                posimg = Sample_prep.intersection(ref.loc[ref['subtype_{}'.format(opt.pdmd)] == 1]['name'].tolist(), allimg)
                for i in negimg:
                    big_images.append([i, level, img_dir + "{}/level{}".format(i, level), 0])
                for i in posimg:
                    big_images.append([i, level, img_dir + "{}/level{}".format(i, level), 1])
            else:
                negimg = Sample_prep.intersection(ref.loc[ref[opt.pdmd] == 0]['name'].tolist(), allimg)
                posimg = Sample_prep.intersection(ref.loc[ref[opt.pdmd] == 1]['name'].tolist(), allimg)
                for i in negimg:
                    big_images.append([i, level, img_dir + "{}/level{}".format(i, level), 0])
                for i in posimg:
                    big_images.append([i, level, img_dir + "{}/level{}".format(i, level), 1])

            datapd = pd.DataFrame(big_images, columns=['slide', 'level', 'path', 'label'])

            test_tiles_list = []
            for idx, row in datapd.iterrows():
                tile_ids = Sample_prep.tile_ids_in(row['slide'], row['level'], row['path'], row['label'])
                test_tiles_list.extend(tile_ids)
            test_tiles = pd.DataFrame(test_tiles_list, columns=['slide', 'level', 'path', 'label'])
            test_tiles.to_csv(data_dir + '/te_sample.csv', header=True, index=False)
            tes = test_tiles
        tecc = len(tes['label'])
        if not os.path.isfile(data_dir + '/test.tfrecords'):
            NYU_loaders.loader(data_dir, 'test')

        m = cnn4.INCEPTION(INPUT_DIM, HYPERPARAMS, meta_graph=opt.modeltoload, log_dir=LOG_DIR,
                           meta_dir=METAGRAPH_DIR, model=opt.mode)
        print("Loaded! Ready for test!")
        if tecc >= bs:
            datasets = data_input2.DataSet(bs, tecc, ep=1, cls=2, mode='test', filename=data_dir + '/test.tfrecords')
            m.inference(datasets, opt.dirr, testset=tes, pmd=opt.pdmd, bs=bs)
        else:
            print("Not enough testing images!")

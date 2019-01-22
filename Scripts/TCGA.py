"""
Join csv/tsv files of TCGA images to create a label list

Created on 1/17/2019

Modified on 1/22/2019 (add a inlist filter so that only images on file are listed)

@author: RH
"""

import pandas as pd
import csv


def flatten(l, a):
    for i in l:
        if isinstance(i, list):
            flatten(i, a)
        else:
            a.append(i)
    return a


image_meta = pd.read_csv('../TCGA_Image_meta.tsv', sep='\t', header=0)
TCGA_list = pd.read_excel('../datafile.S1.1.KeyClinicalData.xls', header=0)

namelist = []
for idx, row in image_meta.iterrows():
    namelist.append(row['File Name'].split('-01Z')[0])

image_meta['bcr_patient_barcode'] = namelist

TCGA_list = TCGA_list.join(image_meta.set_index('bcr_patient_barcode'), on='bcr_patient_barcode')

TCGA_list = TCGA_list.dropna()

labellist = []
for idx, row in TCGA_list.iterrows():
    if row['IntegrativeCluster'] != "Notassigned":
        labellist.append(row['IntegrativeCluster'])
    else:
        labellist.append(row['histology'])

TCGA_list['label'] = labellist

TCGA_list = TCGA_list[TCGA_list.label != 'Mixed']
TCGA_list = TCGA_list[TCGA_list.label != 'Serous']
lbdict = {'CN low': 'Endometroid', 'CN high': 'Serous-like'}
TCGA_list['label'] = TCGA_list['label'].replace(lbdict)

TCGA_list.to_csv('../new_TCGA_list.csv', header=True, index=False)


"""
Tested on 1/22/2019, all images in the previous list are available
"""
# with open('../TCGA_inlist.csv', mode='r') as csv_file:
#     file = csv.reader(csv_file)
#     filenames = flatten(list(file), [])
#
#     TCGA_list_filtered = TCGA_list[TCGA_list['File Name'].isin(filenames)]
#
#     TCGA_list_filtered.to_csv('../filtered_TCGA_list.csv', header=True, index=False)


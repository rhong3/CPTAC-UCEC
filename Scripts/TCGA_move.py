"""
Moving inlist TCGA images to a folder for upload

Created on 1/22/2019

@author: RH
"""

import os
import shutil
import csv

# Get all images in the root directory
def image_ids_in(root_dir, ignore=['.DS_Store', 'dict.csv']):
    ids = []
    for id in os.listdir(root_dir):
        if id in ignore:
            print('Skipping ID:', id)
        else:
            ids.append(id)
    return ids


with open('TCGA_inlist.csv', mode='r') as csv_file:
    filenames = csv.DictReader(csv_file)

imids = image_ids_in('../TCGA-UCEC/')

try:
    os.mkdir('../TCGA-UCEC/inlist')
    os.mkdir('../TCGA-UCEC/outlist')
except(FileExistsError):
    pass

for imid in imids:
    if imid in filenames:
        shutil.move('../TCGA-UCEC/' + str(imid), '../TCGA-UCEC/inlist/' + str(imid))
    else:
        shutil.move('../TCGA-UCEC/' + str(imid), '../TCGA-UCEC/outlist/' + str(imid))



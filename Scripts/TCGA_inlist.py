"""
Finding only TCGA images that in labeled list; prepare for upload to cluster

Created on 1/22/2019

@author: RH
"""

import pandas as pd
import os
import shutil

# Get all images in the root directory
def image_ids_in(root_dir, ignore=['.DS_Store', 'dict.csv']):
    ids = []
    for id in os.listdir(root_dir):
        if id in ignore:
            print('Skipping ID:', id)
        else:
            ids.append(id)
    return ids


TCGA = pd.read_csv('../new_TCGA_list.csv', header=0)
filenames = TCGA['File Name'].tolist()

imids = image_ids_in('../TCGA-UCEC/')

try:
    os.mkdir('../TCGA-UCEC/inlist')
    os.mkdir('../TCGA-UCEC/outlist')
except(FileExistsError):
    pass

inlist_count = 0
outlist_count = 0

for imid in imids:
    if imid in filenames:
        shutil.move('../TCGA-UCEC/' + str(imid), '../TCGA-UCEC/inlist/' + str(imid))
        inlist_count += 1
    else:
        shutil.move('../TCGA-UCEC/' + str(imid), '../TCGA-UCEC/outlist/' + str(imid))
        outlist_count += 1

print(inlist_count)
print(outlist_count)

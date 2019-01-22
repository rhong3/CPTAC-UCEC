"""
Finding only TCGA images that in labeled list; prepare for upload to cluster

Created on 1/22/2019

@author: RH
"""

import pandas as pd
import csv

TCGA = pd.read_csv('../new_TCGA_list.csv', header=0)
filenames = TCGA['File Name'].tolist()

imids = open("../TCGA_images.txt").read().splitlines()
print(imids)

inlist_count = []
outlist_count = []

for imid in imids:
    if imid in filenames:
        inlist_count.append(imid)
    else:
        outlist_count.append(imid)

print(inlist_count)
print(outlist_count)
print(len(inlist_count))
print(len(outlist_count))
csvfile = "../TCGA_inlist.csv"
with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in inlist_count:
        writer.writerow([val])

csvfile = "../TCGA_outlist.csv"
with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in outlist_count:
        writer.writerow([val])
"""
Join csv/tsv files of CPTAC images to create a label list

Created on 11/01/2018

@author: RH
"""

import pandas as pd
import shutil
import os
import csv


def flatten(l, a):
    for i in l:
        if isinstance(i, list):
            flatten(i, a)
        else:
            a.append(i)
    return a


# Get all images in the root directory
def image_ids_in(root_dir, ignore=['.DS_Store', 'dict.csv']):
    ids = []
    for id in os.listdir(root_dir):
        if id in ignore:
            print('Skipping ID:', id)
        else:
            dirname = id.split('_')[-2]
            ids.append((id, dirname))
    return ids


PID = pd.read_csv("../joined_PID.csv", header = 0)
temp = []
ls = []
for idx, row in PID.iterrows():
    if "," in row["Parent Sample ID(s)"]:
        m = row["Parent Sample ID(s)"].split(',')
        for x in m:
            w = row
            ls.append(x)
            temp.append(w)
        PID = PID.drop(idx)
temp = pd.DataFrame(temp)
temp["Parent Sample ID(s)"] = ls
PID = PID.append(temp, ignore_index=True)
PID = PID.sort_values(["Parent Sample ID(s)"], ascending=1)

PID.to_csv("../new_joined_PID.csv", header = True, index = False)

PID = pd.read_csv("../new_joined_PID.csv", header = 0)

ref_list = PID["Parent Sample ID(s)"].tolist()

imids = image_ids_in('../CPTAC_img')
inlist = []
outlist = []
reverse_inlist = []
# try:
#     os.mkdir('../CPTAC_img/inlist')
# except FileExistsError:
#     pass
#
# try:
#     os.mkdir('../CPTAC_img/outlist')
# except FileExistsError:
#     pass

for im in imids:
    if im[1] in ref_list:
        inlist.append(im[0])
        reverse_inlist.append(im[1])
        # shutil.move('../CPTAC_img/'+str(im[0]), '../CPTAC_img/inlist/'+str(im[0]))
    else:
        outlist.append(im[0])
        # shutil.move('../CPTAC_img/' + str(im[0]), '../CPTAC_img/outlist/' + str(im[0]))

csvfile = "../CPTAC_inlist.csv"
with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in inlist:
        writer.writerow([val])

csvfile = "../CPTAC_outlist.csv"
with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in outlist:
        writer.writerow([val])

filtered_PID = PID[PID["Parent Sample ID(s)"].isin(reverse_inlist)]

tpdict = {'CN-High': 'Serous-like', 'CN-Low': 'Endometriod', 'MSI-H': 'MSI', 'POLE': 'POLE', 'Other': 'Other'}
a = filtered_PID['TCGA_subtype']
filtered_PID['Subtype'] = a
filtered_PID.Subtype = filtered_PID.Subtype.replace(tpdict)
filtered_PID = filtered_PID[filtered_PID.Subtype != 'Other']

filtered_PID.to_csv("../filtered_joined_PID.csv", header=True, index=False)

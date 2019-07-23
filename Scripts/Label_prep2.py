"""
Integrated Label Preparation Code

Created on 7/23/2019

@author: RH
"""
import pandas as pd
import shutil
import os
import csv
import numpy as np

# CPTAC
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

imlist = pd.read_excel('../S043_CPTAC_UCEC_Discovery_Cohort_Study_Specimens_r1_Sept2018.xlsx', header=4)
imlist = imlist[imlist['Group'] == 'Tumor ']
cllist = pd.read_csv('../UCEC_V2.1/waffles_updated.txt', sep='\t', header = 0)
cllist = cllist[cllist['Proteomics_Tumor_Normal'] == 'Tumor']

PID = pd.merge(imlist, cllist, how='inner', on=['Participant_ID']) #joined_PID.csv

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
PID = PID.sort_values(["Parent Sample ID(s)"], ascending=1)  #new_joined_PID.csv

ref_list = PID["Parent Sample ID(s)"].tolist()

imids = image_ids_in('../images/CPTAC')
inlist = []
outlist = []
reverse_inlist = []

for im in imids:
    if im[1] in ref_list:
        inlist.append(im[0])
        reverse_inlist.append(im[1])
    else:
        outlist.append(im[0])

filtered_PID = PID[PID["Parent Sample ID(s)"].isin(reverse_inlist)]
filtered_PID = filtered_PID.rename(columns={'TCGA_subtype': 'subtype'})
tpdict = {'CN-High': 'Serous-like', 'CN-Low': 'Endometrioid', 'MSI-H': 'MSI', 'POLE': 'POLE', 'Other': 'Other'}
filtered_PID.subtype = filtered_PID.Subtype.replace(tpdict)
filtered_PID = filtered_PID[filtered_PID.subtype != 'Other']  # filtered_joined_PID.csv

CPTAC = filtered_PID
CPTAC_MUT = pd.read_csv('../UCEC_V2.1/UCEC_CPTAC3_meta_table_V2.1.txt', sep='\t', header=0)
CPTAC_MUTt = CPTAC_MUT[['Participant_ID', 'TP53_TP53', 'TP53_ATM', 'PI3K_PIK3R1', 'PI3K_PIK3CA',
                        'PI3K_PTEN', 'PI3K_MTOR', 'PI3K_PIK3R2', 'PI3K_PPP2R1A', 'HRD_BRCA2', 'JAK1_Mutation']]

CPTAC = CPTAC.join(CPTAC_MUTt.set_index('Participant_ID'), how='inner', on='Participant_ID')

CPTAC = CPTAC.dropna(subset=['TP53_TP53', 'TP53_ATM', 'PI3K_PIK3R1', 'PI3K_PIK3CA', 'PI3K_PTEN',
                             'PI3K_MTOR', 'PI3K_PIK3R2', 'PI3K_PPP2R1A', 'HRD_BRCA2', 'JAK1_Mutation'])

CPTAC = CPTAC.rename(index=str, columns={"TP53_TP53": "TP53", "TP53_ATM": "ATM", 'PI3K_PIK3R1': "PIK3R1",
                                         "PI3K_PIK3CA": "PIK3CA", "PI3K_PTEN": "PTEN", "PI3K_MTOR": "MTOR",
                                         "PI3K_PIK3R2": "PIK3R2", "PI3K_PPP2R1A": "PPP2R1A", "HRD_BRCA2": "BRCA2",
                                         "JAK1_Mutation": "JAK1"}) # MUT_CPTAC_list.csv

CPTAC_lite = CPTAC[['Participant_ID', 'Subtype', 'MSI_status', 'Histologic_type', 'ATM', 'MTOR', 'PIK3R2',
                    'PPP2R1A', 'TP53', 'CTNNB1', 'PIK3CA', 'PIK3R1', 'PTEN', 'BRCA2', 'JAK1']]

CPTAC_lite = CPTAC_lite.rename(index=str, columns={'Participant_ID': 'name',
                                                   'Histologic_type': 'histology', 'MSI_status': 'MSIst'})




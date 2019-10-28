"""
Integrated Label Preparation Code

Created on 7/23/2019

@author: RH
"""
import pandas as pd
import os
import numpy as np
import re


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


# Get all images in the root directory
def image_ids_in_TCGA(root_dir, ignore=['.DS_Store', 'dict.csv']):
    ids = []
    for id in os.listdir(root_dir):
        if id in ignore:
            print('Skipping ID:', id)
        else:
            dirname = re.split('-01Z|-02Z', id)[0]
            sldnum = id.split('-')[5].split('.')[0]
            ids.append((dirname, sldnum))
    ids = pd.DataFrame(ids, columns=['bcr_patient_barcode', 'slide'])

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
filtered_PID = filtered_PID.rename(columns={'TCGA_subtype': 'subtype', 'CTNNB1_Mut': 'CTNNB1'})
tpdict = {'CN-High': 'Serous-like', 'CN-Low': 'Endometrioid', 'MSI-H': 'MSI', 'POLE': 'POLE', 'Other': 'Other'}
filtered_PID.subtype = filtered_PID.subtype.replace(tpdict)
filtered_PID = filtered_PID[filtered_PID.subtype != 'Other']  # filtered_joined_PID.csv

CPTAC = filtered_PID
CPTAC = CPTAC.drop(columns=['MLH1', 'CTNNB1'])
CPTAC_MUT = pd.read_csv('../UCEC_V2.1/UCEC_CPTAC3_meta_table_V2.1_MUT.csv', sep=',', header=0, error_bad_lines=False)
CPTAC_MUTt = CPTAC_MUT[['Participant_ID', 'ARID1A', 'ARID5B', 'ATM', 'BRCA2', 'CTCF', 'CTNNB1', 'EGFR', 'ERBB2',
                        'FBXW7', 'FGFR2', 'JAK1', 'KRAS', 'MLH1', 'MTOR', 'PIK3CA', 'PIK3R1', 'PIK3R2', 'PPP2R1A',
                        'PTEN', 'RPL22', 'TP53', 'FAT1', 'FAT4', 'ZFHX3']]

CPTAC = CPTAC.join(CPTAC_MUTt.set_index('Participant_ID'), how='inner', on='Participant_ID')


CPTAC = CPTAC.dropna(subset=['ARID1A', 'ARID5B', 'ATM', 'BRCA2', 'CTCF', 'CTNNB1', 'EGFR', 'ERBB2',
                        'FBXW7', 'FGFR2', 'JAK1', 'KRAS', 'MLH1', 'MTOR', 'PIK3CA', 'PIK3R1', 'PIK3R2', 'PPP2R1A',
                        'PTEN', 'RPL22', 'TP53', 'FAT1', 'FAT4', 'ZFHX3'])

# MUT_CPTAC_list.csv

CPTAC_lite = CPTAC[['Participant_ID', 'subtype', 'MSI_status', 'Histologic_type', 'ARID1A', 'ARID5B', 'ATM', 'BRCA2',
                    'CTCF', 'CTNNB1', 'EGFR', 'ERBB2', 'FBXW7', 'FGFR2', 'JAK1', 'KRAS', 'MLH1', 'MTOR', 'PIK3CA',
                    'PIK3R1', 'PIK3R2', 'PPP2R1A', 'PTEN', 'RPL22', 'TP53', 'FAT1', 'FAT4', 'ZFHX3']]

CPTAC_lite = CPTAC_lite.rename(index=str, columns={'Participant_ID': 'name',
                                                   'Histologic_type': 'histology', 'MSI_status': 'MSIst'})

# TCGA
image_meta = image_ids_in_TCGA('../images/TCGA/')
TCGA = pd.read_excel('../datafile.S1.1.KeyClinicalData.xls', header=0)
image_meta.to_csv('../TCGA_Images_in_folder.csv', header=0)

unique = list(image_meta.bcr_patient_barcode.unique())

TCGA = TCGA[TCGA['bcr_patient_barcode'].isin(unique)]

# new_TCGA_list.csv

### Subtype
TCGA_list = TCGA[TCGA['IntegrativeCluster'] != "Notassigned"]

TCGA_list = TCGA_list.rename(columns={'IntegrativeCluster': 'subtype', 'bcr_patient_barcode': 'name'})

lbdict = {'CN low': 'Endometrioid', 'CN high': 'Serous-like',
          'UCEC_CN_HIGH': 'Serous-like', 'UCEC_CN_LOW': 'Endometrioid', 'UCEC_MSI': 'MSI', 'UCEC_POLE': 'POLE'}
TCGA_list['subtype'] = TCGA_list['subtype'].replace(lbdict)
TCGA_list = TCGA_list[['name', 'subtype']]

TCGA_MUT = pd.read_csv('../TCGA_MUT_CBP_pan/All_with_mutation_data.tsv', sep='\t', header=0)
TCGA_list_sup = TCGA_MUT.rename(index=str, columns={'Patient ID': 'name', 'Subtype': 'subtype'})
TCGA_list_sup = TCGA_list_sup[['name', 'subtype']]
TCGA_list_sup = TCGA_list_sup[TCGA_list_sup['name'].isin(TCGA['bcr_patient_barcode'].tolist())]
TCGA_list_sup['subtype'] = TCGA_list_sup['subtype'].replace(lbdict)
TCGA_list = TCGA_list.reset_index(drop=True)
TCGA_list_sup = TCGA_list_sup.reset_index(drop=True)
TCGA_list = pd.concat([TCGA_list_sup, TCGA_list])
TCGA_list = TCGA_list.drop_duplicates()

### Mutation
mut_list = TCGA_MUT['Patient ID']
TCGA_mlist = TCGA[TCGA['bcr_patient_barcode'].isin(mut_list)]

for a in ['ARID1A', 'ARID5B', 'ATM', 'BRCA2', 'CTCF', 'CTNNB1', 'EGFR', 'ERBB2', 'FBXW7', 'FGFR2', 'JAK1', 'KRAS',
          'MLH1', 'MTOR', 'PIK3CA', 'PIK3R1', 'PIK3R2', 'PPP2R1A', 'PTEN', 'RPL22', 'TP53', 'FAT1', 'FAT4', 'ZFHX3']:
    cl = pd.read_csv("../TCGA_MUT_CBP_pan/{}.tsv".format(str(a)), sep='\t', header=0)

    cl[a] = 1

    cl.to_csv("../TCGA_MUT_CBP_pan/{}_MUT.csv".format(str(a)), header=True, index=False)

    cll = cl[['Patient ID', a]]

    TCGA_mlist = TCGA_mlist.join(cll.set_index('Patient ID'), how='left', on='bcr_patient_barcode')

    TCGA_mlist[a] = TCGA_mlist[a].fillna(0)

TCGA_mlist = TCGA_mlist.groupby('bcr_patient_barcode').max()

TCGA_mlist.insert(0, 'bcr_patient_barcode', TCGA_mlist.index, allow_duplicates=False)  # MUT_TCGA_list.csv

TCGA_mlist = TCGA_mlist[['bcr_patient_barcode', 'ARID1A', 'ARID5B', 'ATM', 'BRCA2', 'CTCF', 'CTNNB1', 'EGFR', 'ERBB2',
                        'FBXW7', 'FGFR2', 'JAK1', 'KRAS', 'MLH1', 'MTOR', 'PIK3CA', 'PIK3R1', 'PIK3R2', 'PPP2R1A',
                        'PTEN', 'RPL22', 'TP53', 'FAT1', 'FAT4', 'ZFHX3']]

TCGA_mlist = TCGA_mlist.rename(index=str, columns={'bcr_patient_barcode': 'name'})

### Histology
TCGA_hlist = TCGA[TCGA['histology'] != 'Clear cell']
TCGA_hlist = TCGA_hlist.rename(columns={'bcr_patient_barcode': 'name'})
TCGA_hlist = TCGA_hlist[['name', 'histology']]

### MSI
TCGA_slist = TCGA[TCGA['msi_status_7_marker_call'] != 'Indeterminant']
TCGA_slist = TCGA_slist.rename(columns={'bcr_patient_barcode': 'name', 'msi_status_7_marker_call': 'MSIst'})
TCGA_slist = TCGA_slist[['name', 'MSIst']]

TCGA_mg = pd.merge(TCGA_list, TCGA_mlist, how='outer', on='name')
TCGA_mg = pd.merge(TCGA_mg, TCGA_hlist, how='outer', on='name')
TCGA_mg = pd.merge(TCGA_mg, TCGA_slist, how='outer', on='name')

mg = pd.concat([TCGA_mg, CPTAC_lite], sort=True)
ddd = {np.nan: '0NA'}

mg['subtype'] = mg['subtype'].replace(ddd)
mg['histology'] = mg['histology'].replace(ddd)
mg['MSIst'] = mg['MSIst'].replace(ddd)

mg = mg.drop_duplicates()

mgx = pd.get_dummies(mg, columns=['subtype', 'histology', 'MSIst'])
mgx['histology_Endometrioid'] = mgx['histology_Endometrioid'] + mgx['histology_Mixed']
mgx['histology_Serous'] = mgx['histology_Serous'] + mgx['histology_Mixed']
mgx['MSIst_MSS'] = mgx['MSIst_MSS'] + mgx['MSIst_MSI-L']
mgx = mgx.drop(['histology_Clear cell', 'MSIst_MSI-L'], axis=1)
mgx = mgx.set_index('name')
mgx.to_csv('../dummy_His_MUT_joined.csv', header=True, index=True)

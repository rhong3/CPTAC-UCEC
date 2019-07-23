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
                                         "JAK1_Mutation": "JAK1"})  # MUT_CPTAC_list.csv

CPTAC_lite = CPTAC[['Participant_ID', 'Subtype', 'MSI_status', 'Histologic_type', 'ATM', 'MTOR', 'PIK3R2',
                    'PPP2R1A', 'TP53', 'CTNNB1', 'PIK3CA', 'PIK3R1', 'PTEN', 'BRCA2', 'JAK1']]

CPTAC_lite = CPTAC_lite.rename(index=str, columns={'Participant_ID': 'name',
                                                   'Histologic_type': 'histology', 'MSI_status': 'MSIst'})


# TCGA
image_meta = pd.read_csv('../TCGA_Image_meta.tsv', sep='\t', header=0)
TCGA = pd.read_excel('../datafile.S1.1.KeyClinicalData.xls', header=0)

namelist = []
for idx, row in image_meta.iterrows():
    namelist.append(row['File Name'].split('-01Z')[0])

image_meta['bcr_patient_barcode'] = namelist

TCGA = TCGA.join(image_meta.set_index('bcr_patient_barcode'), on='bcr_patient_barcode')

# new_TCGA_list.csv

### Subtype
TCGA_list = TCGA[TCGA['IntegrativeCluster'] != "Notassigned"]

TCGA_list = TCGA_list.rename(columns={'IntegrativeCluster': 'subtype', 'bcr_patient_barcode': 'name'})

lbdict = {'CN low': 'Endometrioid', 'CN high': 'Serous-like'}
TCGA_list['subtype'] = TCGA_list['subtype'].replace(lbdict)
TCGA_list = TCGA_list[['name', 'subtype']]


### Mutation
TCGA_MUT = pd.read_csv('../TCGA_MUT/TCGA_clinical/MUT_clinical.tsv', sep='\t', header=0)

mut_list = TCGA_MUT['submitter_id']
TCGA_mlist = TCGA[TCGA['bcr_patient_barcode'].isin(mut_list)]

for a in ['ATM', 'MTOR', 'PIK3R2', 'PPP2R1A', 'TP53', 'CTNNB1', 'PIK3CA', 'PIK3R1', 'PTEN', 'BRCA2', 'JAK1']:
    cl = pd.read_csv("../TCGA_MUT/TCGA_clinical/{}_MUT_clinical.tsv".format(str(a)), sep='\t', header=0)

    cl[a] = 1

    cl.to_csv("../TCGA_MUT/TCGA_clinical/{}_MUT_clinical.csv".format(str(a)), header=True, index=False)

    cll = cl[['submitter_id', a]]

    TCGA_mlist = TCGA_mlist.join(cll.set_index('submitter_id'), how='left', on='bcr_patient_barcode')

    TCGA_mlist[a] = TCGA_mlist[a].fillna(0)

    pcl = pd.read_csv("../TCGA_MUT/TCGA_clinical/{}_clinical.tsv".format(str(a)), sep='\t', header=0)

    pcl[a] = 1

    pcl.to_csv("../TCGA_MUT/TCGA_clinical/{}_clinical.csv".format(str(a)), header=True, index=False)

    pcll = pcl[['submitter_id', a]]

    TCGA_mlist = pd.merge(TCGA_mlist, pcll,  how='left', left_on=['bcr_patient_barcode'], right_on=['submitter_id'])

    TCGA_mlist[a] = TCGA_mlist[a+'_x'] + TCGA_mlist[a+'_y']

    TCGA_mlist[a] = TCGA_mlist[a]-1

    TCGA_mlist = TCGA_mlist.drop([a+'_x', a+'_y', 'submitter_id'], axis=1)

TCGA_mlist = TCGA_mlist.replace({-1: np.nan})

TCGA_mlist = TCGA_mlist.groupby('bcr_patient_barcode').max()

TCGA_mlist.insert(0, 'bcr_patient_barcode', TCGA_mlist.index, allow_duplicates=False) # MUT_TCGA_list.csv

TCGA_mlist = TCGA_mlist[['bcr_patient_barcode', 'ATM', 'MTOR', 'PIK3R2',
                  'PPP2R1A', 'TP53', 'CTNNB1', 'PIK3CA', 'PIK3R1', 'PTEN', 'BRCA2', 'JAK1']]

TCGA_mlist = TCGA_mlist.rename(index=str, columns={'bcr_patient_barcode': 'name'})

### Histology
TCGA_hlist = TCGA
TCGA_hlist = TCGA_hlist.rename(columns={'bcr_patient_barcode': 'name'})
TCGA_hlist = TCGA_hlist[[['name', 'histology']]]

### MSI
TCGA_slist = TCGA[TCGA['msi_status_7_marker_call'] != 'Indeterminant']
TCGA_slist = TCGA_slist.rename(columns={'bcr_patient_barcode': 'name','msi_status_7_marker_call': 'MSIst'})
TCGA_slist = TCGA_slist[[['name', 'MSIst']]]

TCGA_mg = pd.merge(TCGA_list, TCGA_mlist, how='outer', on='name')
TCGA_mg = pd.merge(TCGA_mg, TCGA_hlist, how='outer', on='name')
TCGA_mg = pd.merge(TCGA_mg, TCGA_slist, how='outer', on='name')

mg = pd.concat([TCGA_mg, CPTAC_lite])
ddd = {np.nan: '0NA'}

mg['subtype'] = mg['subtype'].replace(ddd)
mg['histology'] = mg['histology'].replace(ddd)
mg['MSIst'] = mg['MSIst'].replace(ddd)

mg = mg.drop_duplicates()

mgx = pd.get_dummies(mg, columns=['subtype', 'histology', 'MSIst'])
mgx['histology_Endometrioid'] = mgx['histology_Endometrioid'] + mgx['histology_Mixed']
mgx['histology_Serous'] = mgx['histology_Serous'] + mgx['histology_Mixed']

mgx.to_csv('../dummy_His_MUT_joined.csv', header=True, index=False)
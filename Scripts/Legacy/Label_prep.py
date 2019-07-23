"""
Integrated Label Preparation Code

Created on 4/25/2019

@author: RH
"""
#CPTAC initial prep
import pandas as pd

imlist = pd.read_excel('../S043_CPTAC_UCEC_Discovery_Cohort_Study_Specimens_r1_Sept2018.xlsx', header=4)
imlist = imlist[imlist['Group'] == 'Tumor ']
cllist = pd.read_csv('../UCEC_V2.1/waffles_updated.txt', sep='\t', header = 0)
cllist = cllist[cllist['Proteomics_Tumor_Normal'] == 'Tumor']

joined = pd.merge(imlist, cllist, how='inner', on=['Participant_ID'])

joined.to_csv('../joined_PID.csv', index = False)


#CPTAC prep
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
try:
    os.mkdir('../CPTAC_img/inlist')
except FileExistsError:
    pass

try:
    os.mkdir('../CPTAC_img/outlist')
except FileExistsError:
    pass

for im in imids:
    if im[1] in ref_list:
        inlist.append(im[0])
        reverse_inlist.append(im[1])
        shutil.move('../CPTAC_img/'+str(im[0]), '../CPTAC_img/inlist/'+str(im[0]))
    else:
        outlist.append(im[0])
        shutil.move('../CPTAC_img/' + str(im[0]), '../CPTAC_img/outlist/' + str(im[0]))

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

tpdict = {'CN-High': 'Serous-like', 'CN-Low': 'Endometrioid', 'MSI-H': 'MSI', 'POLE': 'POLE', 'Other': 'Other'}
a = filtered_PID['TCGA_subtype']
filtered_PID['Subtype'] = a
filtered_PID.Subtype = filtered_PID.Subtype.replace(tpdict)
filtered_PID = filtered_PID[filtered_PID.Subtype != 'Other']

filtered_PID.to_csv("../filtered_joined_PID.csv", header=True, index=False)

#TCGA prep
import pandas as pd


def flatten(l, a):
    for i in l:
        if isinstance(i, list):
            flatten(i, a)
        else:
            a.append(i)
    return a


image_meta = pd.read_csv('../TCGA_Image_meta.tsv', sep='\t', header=0)
TCGA_list = pd.read_excel('../TCGA_nature12113-s2/datafile.S1.1.KeyClinicalData.xls', header=0)

namelist = []
for idx, row in image_meta.iterrows():
    namelist.append(row['File Name'].split('-01Z')[0])

image_meta['bcr_patient_barcode'] = namelist

TCGA_list = TCGA_list.join(image_meta.set_index('bcr_patient_barcode'), on='bcr_patient_barcode')

TCGA_list = TCGA_list.dropna()

labellist = []

TCGA_list = TCGA_list[TCGA_list['IntegrativeCluster'] != "Notassigned"]

TCGA_list = TCGA_list.rename(columns={'IntegrativeCluster': 'label'})

lbdict = {'CN low': 'Endometrioid', 'CN high': 'Serous-like'}
TCGA_list['label'] = TCGA_list['label'].replace(lbdict)

TCGA_list.to_csv('../new_TCGA_list.csv', header=True, index=False)


#Mutation prep
import pandas as pd
import numpy as np

CPTAC = pd.read_csv("../filtered_joined_PID.csv", header=0)
CPTAC_MUT = pd.read_csv('../UCEC_V2.1/UCEC_CPTAC3_meta_table_V2.1.txt', sep='\t', header=0)
TCGA = pd.read_csv('../new_TCGA_list.csv', header=0)
TCGA_MUT = pd.read_csv('../TCGA_MUT/TCGA_clinical/MUT_clinical.tsv', sep='\t', header=0)

mut_list = TCGA_MUT['submitter_id']
TCGA = TCGA[TCGA['bcr_patient_barcode'].isin(mut_list)]

for a in ['ATM', 'MTOR', 'PIK3R2', 'PPP2R1A', 'TP53', 'CTNNB1', 'PIK3CA', 'PIK3R1', 'PTEN', 'BRCA2', 'JAK1']:
    cl = pd.read_csv("../TCGA_MUT/TCGA_clinical/{}_MUT_clinical.tsv".format(str(a)), sep='\t', header=0)

    cl[a] = 1

    cl.to_csv("../TCGA_MUT/TCGA_clinical/{}_MUT_clinical.csv".format(str(a)), header=True, index=False)

    cll = cl[['submitter_id', a]]

    TCGA = TCGA.join(cll.set_index('submitter_id'), how='left', on='bcr_patient_barcode')

    TCGA[a] = TCGA[a].fillna(0)

    pcl = pd.read_csv("../TCGA_MUT/TCGA_clinical/{}_clinical.tsv".format(str(a)), sep='\t', header=0)

    pcl[a] = 1

    pcl.to_csv("../TCGA_MUT/TCGA_clinical/{}_clinical.csv".format(str(a)), header=True, index=False)

    pcll = pcl[['submitter_id', a]]

    TCGA = pd.merge(TCGA, pcll,  how='left', left_on=['bcr_patient_barcode'], right_on=['submitter_id'])

    TCGA[a] = TCGA[a+'_x'] + TCGA[a+'_y']

    TCGA[a] = TCGA[a]-1

    TCGA = TCGA.drop([a+'_x', a+'_y', 'submitter_id'], axis=1)

TCGA = TCGA.replace({-1: np.nan})

TCGA = TCGA.groupby('bcr_patient_barcode').max()

TCGA.insert(0, 'bcr_patient_barcode', TCGA.index, allow_duplicates=False)

TCGA.to_csv('../MUT_TCGA_list.csv', header=True, index=False)

CPTAC_MUTt = CPTAC_MUT[['Participant_ID', 'TP53_TP53', 'TP53_ATM', 'PI3K_PIK3R1', 'PI3K_PIK3CA',
                        'PI3K_PTEN', 'PI3K_MTOR', 'PI3K_PIK3R2', 'PI3K_PPP2R1A', 'HRD_BRCA2', 'JAK1_Mutation']]

CPTAC = CPTAC.join(CPTAC_MUTt.set_index('Participant_ID'), how='inner', on='Participant_ID')

CPTAC = CPTAC.dropna(subset=['TP53_TP53', 'TP53_ATM', 'PI3K_PIK3R1', 'PI3K_PIK3CA', 'PI3K_PTEN',
                             'PI3K_MTOR', 'PI3K_PIK3R2', 'PI3K_PPP2R1A', 'HRD_BRCA2', 'JAK1_Mutation'])

CPTAC = CPTAC.rename(index=str, columns={"TP53_TP53": "TP53", "TP53_ATM": "ATM", 'PI3K_PIK3R1': "PIK3R1",
                                         "PI3K_PIK3CA": "PIK3CA", "PI3K_PTEN": "PTEN", "PI3K_MTOR": "MTOR",
                                         "PI3K_PIK3R2": "PIK3R2", "PI3K_PPP2R1A": "PPP2R1A", "HRD_BRCA2": "BRCA2",
                                         "JAK1_Mutation": "JAK1"})

CPTAC.to_csv('../MUT_CPTAC_list.csv', header=True, index=False)

TCGA_lite = TCGA[['bcr_patient_barcode', 'label', 'ATM', 'MTOR', 'PIK3R2',
                  'PPP2R1A', 'TP53', 'CTNNB1', 'PIK3CA', 'PIK3R1', 'PTEN', 'BRCA2', 'JAK1']]

TCGA_lite = TCGA_lite.rename(index=str, columns={'bcr_patient_barcode': 'name', 'label': 'subtype'})

CPTAC_lite = CPTAC[['Participant_ID', 'Subtype', 'ATM', 'MTOR', 'PIK3R2',
                    'PPP2R1A', 'TP53', 'CTNNB1', 'PIK3CA', 'PIK3R1', 'PTEN', 'BRCA2', 'JAK1']]

CPTAC_lite = CPTAC_lite.rename(index=str, columns={'Participant_ID': 'name', 'Subtype': 'subtype'})

jj = pd.concat([TCGA_lite, CPTAC_lite])

ddd = {0: '0NA'}

jj['subtype'] = jj['subtype'].replace(ddd)

jj = jj.drop_duplicates()

jj.to_csv('../MUT_joined.csv', header=True, index=False)

jjx = pd.get_dummies(jj, columns=['subtype'])

jjx.to_csv('../dummy_MUT_joined.csv', header=True, index=False)


#Histology Label prep
import pandas as pd

dummy = pd.read_csv('../dummy_MUT_joined.csv', header=0)

TCGA = pd.read_excel('../datafile.S1.1.KeyClinicalData.xls', header=0)

CPTAC = pd.read_csv('../UCEC_V2.1/UCEC_CPTAC3_meta_table_V2.1.txt', sep='\t', header=0)

TCGA = TCGA[['bcr_patient_barcode', 'histology']]

TCGA = TCGA.rename(index=str, columns={'bcr_patient_barcode': 'name'})

CPTAC = CPTAC[['Participant_ID', 'Histologic_type']]

CPTAC = CPTAC.rename(index=str, columns={'Participant_ID': 'name', 'Histologic_type': 'histology'})

dummyt = dummy[dummy['name'].str.contains("TCGA")]

dummyc = dummy[~dummy['name'].str.contains("TCGA")]

dummyt = dummyt.join(TCGA.set_index('name'), how='left', on='name')

dummyc = dummyc.join(CPTAC.set_index('name'), how='left', on='name')

dummy = pd.concat([dummyt, dummyc])

dummy = dummy.fillna('0NA')

dummy = pd.get_dummies(dummy, columns=['histology'])

dummy = dummy.drop(['histology_Clear cell'], axis=1)

dummy['histology_Endometrioid'] = dummy['histology_Endometrioid'] + dummy['histology_Mixed']

dummy['histology_Serous'] = dummy['histology_Serous'] + dummy['histology_Mixed']

dummy = dummy.replace({'0NA': np.nan})

dummy.to_csv('../dummy_His_MUT_joined.csv', header=True, index=False)

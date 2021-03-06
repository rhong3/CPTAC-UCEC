"""
Prepare labels as CSV dictionaries for fusion

Created on 11/01/2019

@author: RH
"""
import pandas as pd
import numpy as np

labels = pd.read_csv('../dummy_His_MUT_joined.csv', header=0)
CPTAC = pd.read_csv('../CPTAC_clinical.csv', header=0)
CPTAC = CPTAC[CPTAC['Proteomics_Tumor_Normal'] == 'Tumor']
TCGA = pd.read_excel('../datafile.S1.1.KeyClinicalData.xls', header=0)

TCGA = TCGA[['bcr_patient_barcode', 'age', 'BMI']]
TCGA = TCGA.rename(columns={'bcr_patient_barcode': 'name'})
CPTAC = CPTAC[['Participant_ID', 'Age', 'BMI']]
CPTAC = CPTAC.rename(columns={'Participant_ID': 'name', 'Age': 'age'})

clinical = pd.concat([CPTAC, TCGA], ignore_index=True)
clinical = clinical.reset_index()
clinical = clinical[['name', 'age', 'BMI']]
new_labels = labels.merge(clinical, left_on='name', right_on='name')

new_labels = new_labels.replace({"Unknown": np.nan})

new_labels.to_csv('../Fusion_dummy_His_MUT_joined.csv', index=False)


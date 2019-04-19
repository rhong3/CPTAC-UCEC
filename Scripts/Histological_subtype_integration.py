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

dummy = pd.get_dummies(dummy, columns=['histology'])

dummy.to_csv('../dummy_His_MUT_joined.csv', header=True, index=False)


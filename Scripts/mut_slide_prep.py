import pandas as pd

CPTAC = pd.read_csv("../filtered_joined_PID.csv", header=0)
CPTAC_MUT = pd.read_csv('../UCEC_V2.1/UCEC_CPTAC3_meta_table_V2.1.txt', sep='\t', header=0)
TCGA = pd.read_csv('../new_TCGA_list.csv', header=0)
TCGA_MUT = pd.read_csv('../TCGA_MUT/TCGA_clinical/MUT_clinical.tsv', sep='\t', header=0)

mut_list = TCGA_MUT['submitter_id']
TCGA = TCGA[TCGA['bcr_patient_barcode'].isin(mut_list)]

for a in ['ATM', 'TP53', 'CTNNB1', 'PIK3CA', 'PIK3R1', 'PTEN']:
    cl = pd.read_csv("../TCGA_MUT/TCGA_clinical/{}_MUT_clinical.tsv".format(str(a)), sep='\t', header=0)

    cl[a] = 1

    cl.to_csv("../TCGA_MUT/TCGA_clinical/{}_MUT_clinical.csv".format(str(a)), header=True, index=False)

    cll = cl[['submitter_id', a]]

    TCGA = TCGA.join(cll.set_index('submitter_id'), how='left', on='bcr_patient_barcode')

    TCGA = TCGA.fillna(0)

for a in ['ATM', 'TP53', 'CTNNB1', 'PIK3CA', 'PIK3R1', 'PTEN']:
    cl = pd.read_csv("../TCGA_MUT/TCGA_clinical/{}_MUT_clinical.tsv".format(str(a)), sep='\t', header=0)

    cl[a] = 1

    cl.to_csv("../TCGA_MUT/TCGA_clinical/{}_MUT_clinical.csv".format(str(a)), header=True, index=False)

    cll = cl[['submitter_id', a]]

    TCGA = TCGA.join(cll.set_index(['submitter_id', a]), how='outer', rsuffix='rt', on=['bcr_patient_barcode', a])

    TCGA = TCGA.fillna(0)



TCGA = TCGA.groupby('bcr_patient_barcode').max()

TCGA.insert(0, 'bcr_patient_barcode', TCGA.index, allow_duplicates=False)

TCGA.to_csv('../MUT_TCGA_list.csv', header=True, index=False)

CPTAC_MUTt = CPTAC_MUT[['Participant_ID', 'TP53_TP53', 'TP53_ATM', 'PI3K_PIK3R1', 'PI3K_PIK3CA', 'PI3K_PTEN']]

CPTAC = CPTAC.join(CPTAC_MUTt.set_index('Participant_ID'), how='inner', on='Participant_ID')

CPTAC = CPTAC.dropna(subset=['TP53_TP53', 'TP53_ATM', 'PI3K_PIK3R1', 'PI3K_PIK3CA', 'PI3K_PTEN'])

CPTAC = CPTAC.rename(index=str, columns={"TP53_TP53": "TP53", "TP53_ATM": "ATM", 'PI3K_PIK3R1': "PIK3R1", "PI3K_PIK3CA": "PIK3CA", "PI3K_PTEN": "PTEN"})

CPTAC.to_csv('../MUT_CPTAC_list.csv', header=True, index=False)

TCGA_lite = TCGA[['bcr_patient_barcode', 'label', 'ATM', 'TP53', 'CTNNB1', 'PIK3CA', 'PIK3R1', 'PTEN']]

TCGA_lite = TCGA_lite.rename(index=str, columns={'bcr_patient_barcode': 'name', 'label': 'subtype'})

CPTAC_lite = CPTAC[['Participant_ID', 'Subtype', 'ATM', 'TP53', 'CTNNB1', 'PIK3CA', 'PIK3R1', 'PTEN']]

CPTAC_lite = CPTAC_lite.rename(index=str, columns={'Participant_ID': 'name', 'Subtype': 'subtype'})

jj = pd.concat([TCGA_lite, CPTAC_lite])

ddd = {0: '0NA'}

jj['subtype'] = jj['subtype'].replace(ddd)

jj = jj.drop_duplicates()

jj.to_csv('../MUT_joined.csv', header=True, index=False)

jjx = pd.get_dummies(jj, columns=['subtype'])

jjx.to_csv('../dummy_MUT_joined.csv', header=True, index=False)



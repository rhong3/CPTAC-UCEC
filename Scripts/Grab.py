import numpy as np
import pandas as pd

imlist = pd.read_excel('../S043_CPTAC_UCEC_Discovery_Cohort_Study_Specimens_r1_Sept2018.xlsx', header=4)
imlist = imlist[imlist['Group'] == 'Tumor ']
cllist = pd.read_csv('../UCEC_V2.1/waffles_updated.txt', sep='\t', header = 0)
cllist = cllist[cllist['Proteomics_Tumor_Normal'] == 'Tumor']

joined = pd.merge(imlist, cllist, how='inner', on=['Participant_ID'])

joined.to_csv('../joined_PID.csv', index = False)
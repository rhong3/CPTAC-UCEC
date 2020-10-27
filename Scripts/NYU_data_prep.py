import pandas as pd
import numpy as np


# case = pd.read_excel('../NYU/Cases ready for Runyu.xlsx', header=0)
# batch = pd.read_csv('../NYU/Samples_Runyu_Hong_Batch2.csv', header=0)
# case.columns = ['Patient_ID', 'subtype', 'IHC', 'diagnosis']
# batch.columns = ['Slide_ID', 'stain', 'num', 'file']
#
# aa = batch.Slide_ID.str.rsplit('-', n=1, expand=True)
# aa.columns = ['Patient_ID', 'Slide_ID']
# batch['Patient_ID'] = aa['Patient_ID']
# batch['Slide_ID'] = aa['Slide_ID']
# batch = batch[batch['stain'] == 'H&E']
# batch = batch.drop(columns=['stain', 'num'])
#
# aaa = case['diagnosis'].str.split(',', expand=True)
# case['histology'] = aaa[0]
# case['FIGO'] = aaa[1]
#
# case['subtype'] = case['subtype'].replace({'MMR-D': 'MSI', 'CNH': 'CNV-H', 'CNL': 'CNV-L'})
# combined = batch.join(case.set_index('Patient_ID'), on='Patient_ID', how='left')
# combined = combined[['Patient_ID', 'Slide_ID', 'histology', 'subtype', 'FIGO', 'file']]
# combined = combined.dropna(subset=['histology', 'file'])
# combined['file'] = combined['file'].str[:-1]
# combined.to_csv('../NYU/batch2_sum.csv', index=False)


combined = pd.read_csv('../NYU/sum.csv', header=0, usecols=['Patient_ID', 'histology', 'subtype'])
case = pd.read_excel('../NYU/Cases ready for Runyu.xlsx', header=0)
case.columns = ['Patient_ID', 'subtype', 'IHC', 'diagnosis']
case = case[case['Patient_ID'].isin(combined['Patient_ID'].tolist())]
case['subtype_0NA'] = case['subtype'].isna().astype(np.uint8)
case['subtype_Endometrioid'] = case['subtype'].str.contains('CNL').astype(np.uint8)
case['subtype_MSI'] = case['subtype'].str.contains('MMR-D').astype(np.uint8)
case['subtype_Serous-like'] = case['subtype'].str.contains('CNH').astype(np.uint8)
case['subtype_POLE'] = 0
case['TP53'] = case['IHC'].str.contains('p53 overexpressed').astype(np.uint8)
case['histology_Endometrioid'] = case['diagnosis'].str.contains('Endometrioid', case=False).astype(np.uint8)
case['histology_Serous'] = case['diagnosis'].str.contains('Serous', case=False).astype(np.uint8)
case['histology_Mixed'] = np.abs(case['histology_Endometrioid']
                                 +case['histology_Serous']-1).astype(np.bool).astype(np.uint8)

case = case[['Patient_ID', 'subtype_0NA', 'subtype_Endometrioid', 'subtype_MSI', 'subtype_Serous-like', 'subtype_POLE',
             'TP53', 'histology_Endometrioid', 'histology_Serous', 'histology_Mixed']]
case.columns = ['name', 'subtype_0NA', 'subtype_Endometrioid', 'subtype_MSI', 'subtype_Serous-like', 'subtype_POLE',
                'TP53', 'histology_Endometrioid', 'histology_Serous', 'histology_Mixed']
case['age'] = np.nan
case['BMI'] = np.nan
case.to_csv('../NYU/label.csv', index=False)









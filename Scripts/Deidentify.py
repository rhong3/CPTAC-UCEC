# Deidentify NYU data
import pandas as pd
import os

case = pd.read_excel('../NYU/Cases ready for Runyu.xlsx', header=0)
dicts = dict(zip(case['Case'].tolist(), case['NYU_name'].tolist()))

# for n in ['sum.csv', 'batch2_sum.csv', 'batch1_sum.csv']:
#     sum = pd.read_csv('../NYU/'+n, header=0)
#     sum['Patient_ID'] = sum['Patient_ID'].replace(dicts)
#     sum.to_csv('../NYU/'+n, index=False)

# case = pd.read_excel('../NYU/Cases ready for Runyu.xlsx', header=0)
# case['NYU_name'] = case['Case'].map(dicts)
# case = case[['Case', 'NYU_name', 'Group', 'IHC', 'Diagnosis']]
# case.to_excel('../NYU/Cases ready for Runyu.xlsx', index=False)

for n in ['Samples_Runyu_Hong_Batch3.csv']:
    sum = pd.read_csv('../NYU/'+n, header=0)
    sum['Slide ID'] = sum['Slide ID'].replace(dicts, regex=True)
    sum.to_csv('../NYU/'+n, index=False)

# label = pd.read_csv('../NYU/label.csv', header=0)
# label['name'] = label['name'].replace(dicts)
# label.to_csv('../NYU/label.csv', index=False)

# for m in os.listdir('../tiles/'):
#     if m in dicts.keys():
#         os.rename('../tiles/'+m, '../tiles/'+dicts[m])


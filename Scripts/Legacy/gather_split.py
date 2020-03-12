"""
Get baseline models data split

Created on 11/15/2019

@author: RH
"""
import pandas as pd
import numpy as np

flist = ['ATM-S1981', 'ATR-T1989', 'LINE1_ORF1p', 'NBN-S343', 'RAD50-S635']
for i in flist:
    tr = pd.read_csv('../Results/X1{}/data/tr_sample.csv'.format(i), header=0)
    te = pd.read_csv('../Results/X1{}/data/te_sample.csv'.format(i), header=0)
    va = pd.read_csv('../Results/X1{}/data/va_sample.csv'.format(i), header=0)
    trunq = list(tr.slide.unique())
    teunq = list(te.slide.unique())
    vaunq = list(va.slide.unique())
    tepd = pd.DataFrame(columns=['slide', 'set'])
    tepd['slide'] = teunq
    tepd['set'] = 'test'
    trpd = pd.DataFrame(columns=['slide', 'set'])
    trpd['slide'] = trunq
    trpd['set'] = 'train'
    vapd = pd.DataFrame(columns=['slide', 'set'])
    vapd['slide'] = vaunq
    vapd['set'] = 'validation'

    pdpd = pd.concat([trpd, vapd, tepd], ignore_index=True)
    pdpd.columns = ['slide', 'set']
    pdpd.to_csv('../split/{}.csv'.format(i), index=False)

# # For TP53-244 split
# tr = pd.read_csv('../Results/NL5/X1TP53-244/data/tr_sample.csv', header=0)
# te = pd.read_csv('../Results/NL5/X1TP53-244/data/te_sample.csv', header=0)
# va = pd.read_csv('../Results/NL5/X1TP53-244/data/va_sample.csv', header=0)
# trunq = list(tr.slide.unique())
# teunq = list(te.slide.unique())
# vaunq = list(va.slide.unique())
# tepd = pd.DataFrame(columns=['slide', 'set'])
# tepd['slide'] = teunq
# tepd['set'] = 'test'
# trpd = pd.DataFrame(columns=['slide', 'set'])
# trpd['slide'] = trunq
# trpd['set'] = 'train'
# vapd = pd.DataFrame(columns=['slide', 'set'])
# vapd['slide'] = vaunq
# vapd['set'] = 'validation'
# pdpd = pd.concat([trpd, vapd, tepd], ignore_index=True)
# pdpd.columns = ['slide', 'set']
#
# ref = pd.read_csv('../dummy_His_MUT_joined.csv', header=0)
# ref = ref[~ref['TP53'].isna()]
#
# pdpd = pdpd[pdpd['slide'].isin(ref['name'].tolist())]
#
# ref = ref[~ref['name'].isin(pdpd['slide'].tolist())]['name'].tolist()
#
# pdlst = []
# for m in ref:
#     ppp = np.random.random()
#     print(ppp)
#     if ppp < 0.1:
#         pdlst.append([m, 'test'])
#     elif ppp > 0.9:
#         pdlst.append([m, 'validation'])
#     else:
#         pdlst.append([m, 'train'])
#
# pdpdd = pd.DataFrame(pdlst, columns=['slide', 'set'])
# pdpd = pdpd.append(pdpdd)
# pdpd.to_csv('../split/TP53.csv', index=False)

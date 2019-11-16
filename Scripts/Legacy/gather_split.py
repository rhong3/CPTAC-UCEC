"""
Get baseline models data split

Created on 11/15/2019

@author: RH
"""
import pandas as pd

flist = ['his', 'MSIst', 'ST', 'ARID1A', 'ATM', 'BRCA2', 'CTCF', 'CTNNB1', 'FAT1', 'FBXW7', 'FGFR2', 'JAK1', 'KRAS',
         'MTOR', 'PIK3CA', 'PIK3R1', 'PPP2R1A', 'PTEN', 'RPL22', 'TP53', 'ZFHX3']
for i in flist:
    tr = pd.read_csv('../Results/NL5/I1{}/data/tr_sample.csv'.format(i), header=0)
    te = pd.read_csv('../Results/NL5/I1{}/data/te_sample.csv'.format(i), header=0)
    va = pd.read_csv('../Results/NL5/I1{}/data/va_sample.csv'.format(i), header=0)
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

"""
Count images and patients

Created on 11/01/2019

@author: RH
"""
import os
import pandas as pd

ref = pd.read_csv('../dummy_His_MUT_joined.csv', header=0)
refls = ref['name'].tolist()
subfolders = [f.name for f in os.scandir('../tiles/') if f.is_dir()]
for w in refls:
    if w not in subfolders:
        print(w)

TCGA = 0
CPTAC = 0
sumlist = []
for m in os.listdir('../tiles/'):
    print(m)
    summ = 0
    if m not in refls:
        print('Not In: ', m)
    try:
        for n in os.listdir('../tiles/{}/level0'.format(m)):
            if '.csv' in n:
                summ += 1
                if 'TCGA' in m:
                    TCGA += 1
                else:
                    CPTAC += 1
        sumlist.append([m, summ])
    except FileNotFoundError:
        print('Error: ', m)
print(TCGA)
print(CPTAC)
print(TCGA+CPTAC)

sumlist = pd.DataFrame(sumlist, columns=["patient", "num_of_slides"])
sumlist.to_csv("../patient_slides_count.csv", index=False)

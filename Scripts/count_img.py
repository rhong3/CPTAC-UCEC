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
for m in os.listdir('../tiles/'):
    print(m)
    if m not in refls:
        print('Not In: {}'.format(m))
    try:
        for n in os.listdir('../tiles/{}/level0'.format(m)):
            if '.csv' in n:
                if 'TCGA' in m:
                    TCGA += 1
                else:
                    CPTAC += 1
    except FileNotFoundError:
        print('Error: {}'.format(m))
print(TCGA)
print(CPTAC)
print(TCGA+CPTAC)


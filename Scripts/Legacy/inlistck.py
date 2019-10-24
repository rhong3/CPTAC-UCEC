import pandas as pd
import os


ref = pd.read_csv('../dummy_His_MUT_joined.csv', header=0)
refls = ref['name'].tolist()
subfolders = [f.name for f in os.scandir('../tiles/') if f.is_dir()]
for w in refls:
    if w not in subfolders:
        print(w)

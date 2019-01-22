"""
Join csv/tsv files of CPTAC images to create a label list

Created on 11/01/2018

@author: RH
"""

import pandas as pd

PID = pd.read_csv("../joined_PID.csv", header = 0)
temp = []
ls = []
for idx, row in PID.iterrows():
    if "," in row["Parent Sample ID(s)"]:
        m = row["Parent Sample ID(s)"].split(',')
        for x in m:
            w = row
            ls.append(x)
            temp.append(w)
        PID = PID.drop(idx)
temp = pd.DataFrame(temp)
temp["Parent Sample ID(s)"] = ls
PID = PID.append(temp, ignore_index=True)
PID = PID.sort_values(["Parent Sample ID(s)"], ascending=1)

PID.to_csv("../new_joined_PID.csv", header = True, index = False)

PID = pd.read_csv("../new_joined_PID.csv", header = 0)

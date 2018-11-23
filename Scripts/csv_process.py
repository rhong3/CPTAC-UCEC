import numpy as np
import pandas as pd

PID = pd.read_csv("../joined_PID.csv", header = 0)

for idx, row in PID.iterrows():
    if "," in row["Parent Sample ID(s)"]:
        m = row["Parent Sample ID(s)"].split(',')
        print(m)
        for x in m:
            w = row
            w["Parent Sample ID(s)"] = x
            PID.append(w)
        PID.drop([idx])

print(PID["Parent Sample ID(s)"])


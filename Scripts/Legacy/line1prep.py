import pandas as pd

labels = pd.read_csv('../Fusion_dummy_His_MUT_joined.csv', header=0)
line = pd.read_csv('../Line1.csv', header=0)

line['name'] = line['Proteomics_Participant_ID']
line = line.drop(['Proteomics_Participant_ID', 'Histologic_type', 'Genomics_subtype', 'TP53_TP53'], axis=1)
labels = labels.join(line.set_index('name'), on='name')
labels['LINE1_ORF1p'] = (labels['LINE1_ORF1p'].dropna() > 0).astype(int)
labels['RAD50-S635'] = (labels['RAD50-S635'].dropna() > 0).astype(int)
labels['NBN-S343'] = (labels['NBN-S343'].dropna() > 0).astype(int)
labels['ATR-T1989'] = (labels['ATR-T1989'].dropna() > 0).astype(int)
labels['ATM-S1981'] = (labels['ATM-S1981'].dropna() > 0).astype(int)

labels.to_csv('../Fusion_dummy_His_MUT_joined.csv', index=False)

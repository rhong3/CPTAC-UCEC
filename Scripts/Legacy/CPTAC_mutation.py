import pandas as pd

mut = pd.read_csv('../UCEC_V2.1/UCEC_somatic_mutation_gene_level_V2.1.csv', header=0)
CPTAC = pd.read_csv('../UCEC_V2.1/UCEC_CPTAC3_meta_table_V2.1.csv', sep=',', header=0, error_bad_lines=False)

mut = mut.T
mut.columns = mut.iloc[0]
mut = mut[1:]

CPTAC = CPTAC.drop(columns=['MLH1', 'MSH6', 'PMS2'])
CPTAC = CPTAC.join(mut, how='inner', on='idx')

CPTAC.to_csv('../UCEC_V2.1/UCEC_CPTAC3_meta_table_V2.1_MUT.csv', index=False)


# Find common test samples
a = 'CNVH'
b = 'PTEN'
al = read.table(paste('/Users/rh2740/documents/CPTAC-UCEC/split/', a, '.csv', sep=''), header=T,sep=',')
bl = read.table(paste('/Users/rh2740/documents/CPTAC-UCEC/split/', b, '.csv', sep=''), header=T,sep=',')
ref = read.table('/Users/rh2740/documents/CPTAC-UCEC/Fusion_dummy_His_MUT_joined.csv', header=T,sep=',')
al = al[al['set'] == 'test', ]
bl = bl[bl['set'] == 'test', ]
common = intersect(al$slide, bl$slide)  
ref.x = ref[ref$name %in% common, ]

ref.mix = ref[ref$histology_Mixed == 1, ]

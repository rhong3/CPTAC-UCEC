# Make tables
his = read.csv("~/documents/CPTAC-UCEC/Results/Statistics_histology.csv")
his = his[his$Tiles == "NL5", ]
his = his[his$Architecture != "I4", ]
his$Feature = "Histology"
his.ss = his[,c(1,3,4,5, 27,28,29,51)]
his.ss[,-c(1,8)] = round(his.ss[,-c(1,8)], 3)
# his.ss$Patient_ROC = paste(his.ss$Patient_ROC, ' (', his.ss$Patient_ROC.95.CI_lower, '-', his.ss$Patient_ROC.95.CI_upper, ')', sep='')
# his.ss$Tile_ROC = paste(his.ss$Tile_ROC, ' (', his.ss$Tile_ROC.95.CI_lower, '-', his.ss$Tile_ROC.95.CI_upper, ')', sep='')
his.df = his.ss[, c(1,3,6,8)]
his.df = his.df[with(his.df , order(Feature, Architecture)), ]


MSI = read.csv("~/documents/CPTAC-UCEC/Results/Statistics_MSI.csv")
MSI = MSI[MSI$Tiles == "NL5", ]
MSI = MSI[MSI$Architecture != "I4", ]
MSI$Feature = "MSI-High"
MSI.ss = MSI[,c(1,3,4,5, 27,28,29,51)]
MSI.ss[,-c(1,8)] = round(MSI.ss[,-c(1,8)], 3)
# MSI.ss$Patient_ROC = paste(MSI.ss$Patient_ROC, ' (', MSI.ss$Patient_ROC.95.CI_lower, '-', MSI.ss$Patient_ROC.95.CI_upper, ')', sep='')
# MSI.ss$Tile_ROC = paste(MSI.ss$Tile_ROC, ' (', MSI.ss$Tile_ROC.95.CI_lower, '-', MSI.ss$Tile_ROC.95.CI_upper, ')', sep='')
MSI.df = MSI.ss[, c(1,3,6,8)]
MSI.df = MSI.df[with(MSI.df, order(Feature, Architecture)), ]


CNVH = read.csv("~/documents/CPTAC-UCEC/Results/Statistics_special.csv")
CNVH = CNVH[CNVH$Tiles == "NL5", ]
CNVH = CNVH[CNVH$Architecture != "I4", ]
CNVH.ss = CNVH[,c(1,2,5,6,7,29,30,31)]
CNVH.ss[,-c(1,2)] = round(CNVH.ss[,-c(1,2)], 3)
# CNVH.ss$Patient_ROC = paste(CNVH.ss$Patient_ROC, ' (', CNVH.ss$Patient_ROC.95.CI_lower, '-', CNVH.ss$Patient_ROC.95.CI_upper, ')', sep='')
# CNVH.ss$Tile_ROC = paste(CNVH.ss$Tile_ROC, ' (', CNVH.ss$Tile_ROC.95.CI_lower, '-', CNVH.ss$Tile_ROC.95.CI_upper, ')', sep='')
CNVH.df = CNVH.ss[, c(1,2,4,7)]
CNVH.df = CNVH.df[with(CNVH.df, order(Feature, Architecture)), ]


mut = read.csv("~/documents/CPTAC-UCEC/Results/Statistics_mutations.csv")
mut = mut[mut$Tiles == "NL5", ]
mut = mut[mut$Architecture != "I4", ]
mut.ss = mut[,c(1,2,4,5,6,28,29,30)]
mut.ss[,-c(1,2)] = round(mut.ss[,-c(1,2)], 3)
# mut.ss$Patient_ROC = paste(mut.ss$Patient_ROC, ' (', mut.ss$Patient_ROC.95.CI_lower, '-', mut.ss$Patient_ROC.95.CI_upper, ')', sep='')
# mut.ss$Tile_ROC = paste(mut.ss$Tile_ROC, ' (', mut.ss$Tile_ROC.95.CI_lower, '-', mut.ss$Tile_ROC.95.CI_upper, ')', sep='')
mut.df = mut.ss[, c(1,2,4,7)]
mut.df = mut.df[mut.df$Mutation != 'TP53-old', ]
mut.df = mut.df[mut.df$Mutation != 'TP53-244', ]
mut.df = mut.df[mut.df$Mutation != 'ARID1A-sp', ]
mut.df = mut.df[mut.df$Mutation != 'ARID1A-NF', ]
mut.df = mut.df[mut.df$Architecture != 'R1', ]
colnames(mut.df)[1] = 'Feature'
mut.df = mut.df[with(mut.df, order(Feature, Architecture)), ]

row.names(his.df) <- NULL
row.names(CNVH.df) <- NULL
row.names(MSI.df) <- NULL
row.names(mut.df) <- NULL

df = rbind(his.df, CNVH.df, MSI.df, mut.df)
df$Feature = gsub("SL", "CNV-H from endometrioid", df$Feature)
df$Feature = gsub("CNVH", "CNV-H (Serous-like) binary", df$Feature)
df$Architecture = gsub("I5", "IR1", df$Architecture)
df$Architecture = gsub("I6", "IR2", df$Architecture)
df$Architecture = gsub("X1", "P2", df$Architecture)
df$Architecture = gsub("X2", "P1", df$Architecture)
df$Architecture = gsub("X3", "P4", df$Architecture)
df$Architecture = gsub("X4", "P3", df$Architecture)
df$Architecture = gsub("F1", "PC2", df$Architecture)
df$Architecture = gsub("F2", "PC1", df$Architecture)
df$Architecture = gsub("F3", "PC4", df$Architecture)
df$Architecture = gsub("F4", "PC3", df$Architecture)
df = df[,c(4,1,2,3)]
colnames(df) = c('Feature', 'Architecture', 'Per-patient AUROC', 'Per-tile AUROC')
# write.csv(df, "~/documents/CPTAC-UCEC/Results/NL5/Results_table.csv", row.names=FALSE)

arch = c('I1', 'I2', 'I3', 'IR1', 'IR2', 'P1', 'P2', 'P3', 'P4', 'PC1', 'PC2', 'PC3', 'PC4')
outdf_patient = data.frame(matrix("", ncol = 13, nrow = 22))
rownames(outdf_patient) = c("Histology", "CNV-H (Serous-like) binary", "CNV-H from endometrioid", "MSI-High", "ARID1A", "ATM",   
                            "BRCA2", "CTCF",  "CTNNB1", "FAT1",  "FBXW7", "FGFR2", "JAK1",  "KRAS",  
                            "MTOR", "PIK3CA", "PIK3R1", "PPP2R1A", "PTEN",  "RPL22", "TP53",  "ZFHX3")
colnames(outdf_patient) = arch

outdf_tile = data.frame(matrix("", ncol = 13, nrow = 22))
rownames(outdf_tile) = c("Histology", "CNV-H (Serous-like) binary", "CNV-H from endometrioid", "MSI-High", "ARID1A", "ATM",   
                            "BRCA2", "CTCF",  "CTNNB1", "FAT1",  "FBXW7", "FGFR2", "JAK1",  "KRAS",  
                            "MTOR", "PIK3CA", "PIK3R1", "PPP2R1A", "PTEN",  "RPL22", "TP53",  "ZFHX3")
colnames(outdf_tile) = arch

for (aa in arch){
  sp = df[df$Architecture == aa,]
  outdf_patient[aa] = sp[,3]
  outdf_tile[aa] = sp[,4]
}

write.csv(outdf_patient, "~/documents/CPTAC-UCEC/Results/NL5/Results_Matrix_patient.csv")
write.csv(outdf_tile, "~/documents/CPTAC-UCEC/Results/NL5/Results_Matrix_tile.csv")

library(Hmisc)
library(Rfast)
library(dplyr)
library(pheatmap)
myBreaks <- seq(0.5, 1, by=0.005)
# myBreaks <- seq(0, 1, by=0.01)
pdf(file="~/documents/CPTAC-UCEC/Results/NL5/Results_Matrix_patient.pdf",
    width=6.5,height=7)
pheatmap(outdf_patient, cluster_cols = FALSE, cluster_rows = FALSE, main = 'Per-patient AUROC', breaks = myBreaks)
dev.off()

pdf(file="~/documents/CPTAC-UCEC/Results/NL5/Results_Matrix_tile.pdf",
    width=6.5,height=7)
pheatmap(outdf_tile, cluster_cols = FALSE, cluster_rows = FALSE, main = 'Per-tile AUROC', breaks = myBreaks)
dev.off()

# box plot Mann-Whitney U test (Fig.3 in Lung paper)
# https://www.r-bloggers.com/add-p-values-and-significance-levels-to-ggplots/
library(ggplot2)
library(ggpubr)
features = c('his', 'MSIst', 'FAT1', 'TP53', 'PTEN', 'ZFHX3', 'ARID1A', 'ATM', 'BRCA2', 'CTCF', 'CTNNB1', 'FBXW7', 'JAK1', 'KRAS', 'MTOR', 'PIK3CA', 'PIK3R1', 'PPP2R1A', 'RPL22', 'FGFR2', 'SL', 'CNVH')
arch = c('I5', 'I6', 'X2', 'X1', 'X4', 'X3', 'F2', 'F1', 'F4', 'F3', 'I1', 'I2', 'I3')

F = 'his'
a = 'X1'
i = paste(a, F, sep="")

if (f == 'his'){
  pos = "Serous_score"
  lev = c('Endometrioid', 'Serous')  
} else if(f == 'MSIst'){
  pos = "MSI.H_score"
  lev = c('MSS', 'MSI-H')
} else if(f == 'SL' | f == 'CNVH'){
  pos = "POS_score"
  lev = c('negative', 'Serous-like')  
} else{
  pos = "POS_score"
  lev = c('negative', f)
}

pdf(file="~/documents/CPTAC-UCEC/Results/test.pdf",
    width=14,height=7)
Test_tile <- read.csv(paste("~/documents/CPTAC-UCEC/Results/NL5/", i, "/out/Test_tile.csv", sep=''))
ggboxplot(Test_tile, x = "True_label", y = pos,
               color = "True_label", palette = "jco",
               add = "jitter")+ stat_compare_means()
dev.off()



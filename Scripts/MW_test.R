# box plot Wilcoxon test (Fig.3 in Lung paper)
# https://www.r-bloggers.com/add-p-values-and-significance-levels-to-ggplots/
library(ggplot2)
library(ggpubr)
features = c('his', 'MSIst', 'FAT1', 'TP53', 'PTEN', 'ZFHX3', 'ARID1A', 'ATM', 'BRCA2', 'CTCF', 'CTNNB1', 'FBXW7', 'JAK1', 'KRAS', 'MTOR', 'PIK3CA', 'PIK3R1', 'PPP2R1A', 'RPL22', 'FGFR2', 'SL', 'CNVH')
arch = c('I5', 'I6', 'X2', 'X1', 'X4', 'X3', 'F2', 'F1', 'F4', 'F3', 'I1', 'I2', 'I3')

for (a in arch){
  tile_all = data.frame(Prediction_score= numeric(0), True_label= character(0), feature = character(0))
  for (f in features){
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
    i = paste(a, f, sep="")
    Test_tile <- read.csv(paste("~/documents/CPTAC-UCEC/Results/NL5/", i, "/out/Test_tile.csv", sep=''))
    Test_tile = Test_tile[, c(pos, "True_label")]
    Test_tile['feature'] = f
    levels(Test_tile$True_label) <- c(levels(Test_tile$True_label), 'negative', 'positive')
    Test_tile$True_label[Test_tile$True_label==lev[1]] = 'negative'
    Test_tile$True_label[Test_tile$True_label==lev[2]] = 'positive'
    colnames(Test_tile) = c('Prediction_score', 'True_label', 'feature')
    tile_all = rbind(tile_all, Test_tile)
  }

  pp = ggboxplot(tile_all, x = "feature", y = "Prediction_score",
            color = "black", fill = "True_label", palette = "grey")+ 
    stat_compare_means(method.args = list(alternative = "greater"), aes(group = True_label), label = "p.signif", label.y = 1.1) + 
    stat_compare_means(method.args = list(alternative = "greater"), aes(group = True_label), label = "p.format", label.y = 1.15)
  
  pdf(file=paste("~/documents/CPTAC-UCEC/Results/Wilcoxon/", a, ".pdf", sep=''),
      width=28,height=7)
  grid.arrange(pp,nrow=1)
  dev.off()
}



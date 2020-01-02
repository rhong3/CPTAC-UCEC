# box plot Wilcoxon test (Fig.3 in Lung paper)
# https://www.r-bloggers.com/add-p-values-and-significance-levels-to-ggplots/
library(ggplot2)
library(ggpubr)
features = c('his', 'SL', 'CNVH', 'TP53', 'FAT1', 'MSIst', 'ZFHX3', 'PTEN', 'FGFR2', 'MTOR', 'CTCF', 'PIK3R1', 'PIK3CA', 'ARID1A', 'JAK1', 'CTNNB1', 'KRAS', 'FBXW7', 'RPL22', 'BRCA2')
arch = c('I5', 'I6', 'X2', 'X1', 'X4', 'X3', 'F2', 'F1', 'F4', 'F3', 'I1', 'I2', 'I3')

for (a in arch){
  tile_all = data.frame(Prediction_score= numeric(0), True_label= character(0), feature = character(0))
  for (f in features){
    if (f == 'his'){
      pos = "Serous_score"
      lev = c('Endometrioid', 'Serous')
      mm = 'Histology'
    } else if(f == 'MSIst'){
      pos = "MSI.H_score"
      lev = c('MSS', 'MSI-H')
      mm = 'MSI-high'
    } else if(f == 'SL'){
      pos = "POS_score"
      lev = c('negative', 'Serous-like')
      mm = "CNV-H (endometrioid)"
    } else if(f == 'CNVH'){
      pos = "POS_score"
      lev = c('negative', 'Serous-like')
      mm = "CNV-H"
    } else{
      pos = "POS_score"
      lev = c('negative', f)
      mm = f
    }
    i = paste(a, f, sep="")
    Test_tile <- read.csv(paste("~/documents/CPTAC-UCEC/Results/NL5/", i, "/out/Test_tile.csv", sep=''))
    Test_tile = Test_tile[, c(pos, "True_label")]
    Test_tile['feature'] = mm
    levels(Test_tile$True_label) <- c(levels(Test_tile$True_label), 'negative', 'positive')
    Test_tile$True_label[Test_tile$True_label==lev[1]] = 'negative'
    Test_tile$True_label[Test_tile$True_label==lev[2]] = 'positive'
    colnames(Test_tile) = c('Prediction_score', 'True_label', 'feature')
    tile_all = rbind(tile_all, Test_tile)
  }

  pp = ggboxplot(tile_all, x = "feature", y = "Prediction_score",
            color = "black", fill = "True_label", palette = "grey")+ 
    stat_compare_means(method.args = list(alternative = "greater"), aes(group = True_label), label = "p.signif", label.y = 1.1) + 
    stat_compare_means(method.args = list(alternative = "greater"), aes(group = True_label), label = "p.format", label.y = 1.15)+ 
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  pdf(file=paste("~/documents/CPTAC-UCEC/Results/Wilcoxon/1-tail/", a, ".pdf", sep=''),
      width=18.5,height=8)
  grid.arrange(pp,nrow=1)
  dev.off()
}


## For figure
todolist = list(c('X1', 'his'), c('X2', 'SL'), c('X3', 'CNVH'), c('X1', 'TP53'), c('F1', 'FAT1'), c('I5', 'MSIst'), c('I5', 'ZFHX3'), c('I2', 'PTEN'), c('F3', 'FGFR2'),
             c('X2', 'MTOR'), c('X3', 'CTCF'), c('I5', 'PIK3R1'), c('X3', 'PIK3CA'), c('I6', 'ARID1A'), c('F1', 'JAK1'), c('I6', 'CTNNB1'), c('F1', 'KRAS'), 
             c('I3', 'FBXW7'), c('I3', 'RPL22'), c('I5', 'BRCA2'))
tile_all = data.frame(Prediction_score= numeric(0), True_label= character(0), feature = character(0))
for (f in todolist){
  if (f[2] == 'his'){
    pos = "Serous_score"
    lev = c('Endometrioid', 'Serous')
    mm = 'Histology'
  } else if(f[2] == 'MSIst'){
    pos = "MSI.H_score"
    lev = c('MSS', 'MSI-H')
    mm = 'MSI-high'
  } else if(f[2] == 'SL'){
    pos = "POS_score"
    lev = c('negative', 'Serous-like')
    mm = "CNV-H (endometrioid)"
  } else if(f[2] == 'CNVH'){
    pos = "POS_score"
    lev = c('negative', 'Serous-like')
    mm = "CNV-H"
  } else{
    pos = "POS_score"
    lev = c('negative', f[2])
    mm = f[2]
  }
  i = paste(f[1], f[2], sep="")
  Test_tile <- read.csv(paste("~/documents/CPTAC-UCEC/Results/NL5/", i, "/out/Test_tile.csv", sep=''))
  Test_tile = Test_tile[, c(pos, "True_label")]
  Test_tile['feature'] = mm
  levels(Test_tile$True_label) <- c(levels(Test_tile$True_label), 'negative', 'positive')
  Test_tile$True_label[Test_tile$True_label==lev[1]] = 'negative'
  Test_tile$True_label[Test_tile$True_label==lev[2]] = 'positive'
  colnames(Test_tile) = c('Prediction_score', 'True_label', 'feature')
  tile_all = rbind(tile_all, Test_tile)
}

pp = ggboxplot(tile_all, x = "feature", y = "Prediction_score",
               color = "black", fill = "True_label", palette = "grey")+ 
  stat_compare_means(method.args = list(alternative = "greater"), aes(group = True_label), label = "p.signif", label.y = 1.1) + 
  stat_compare_means(method.args = list(alternative = "greater"), aes(group = True_label), label = "p.format", label.y = 1.15) + 
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

pdf(file=paste("~/documents/CPTAC-UCEC/Results/Wilcoxon/Figure.pdf", sep=''),
    width=18.5,height=8)
grid.arrange(pp,nrow=1)
dev.off()


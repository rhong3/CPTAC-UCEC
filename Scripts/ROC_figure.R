# ROC for figure
pdf(file=paste("~/documents/CPTAC-UCEC/Manuscript/Figures/ROC/ROC_slide.pdf", sep=''),
    width=5,height=5)
library(pROC)
# plot.new()
todolist = list(c('X1', 'his'), c('X2', 'SL'), c('X3', 'CNVH'), c('X1', 'TP53'))
color = c("#55DDE0", "#33658A", "#F6AE2D", "#F26419")
for (f in todolist){
  if (f[2] == 'his'){
    pos = "Serous_score"
    lev = c('Endometrioid', 'Serous')
    mm = 'Histology'
    sw = FALSE
    clr = "#55DDE0"
  } else if(f[2] == 'MSIst'){
    pos = "MSI.H_score"
    lev = c('MSS', 'MSI-H')
    mm = 'MSI-high'
    sw = TRUE
  } else if(f[2] == 'SL'){
    pos = "POS_score"
    lev = c('negative', 'Serous-like')
    mm = "CNV-H (endometrioid)"
    sw = TRUE
    clr = "#33658A"
  } else if(f[2] == 'CNVH'){
    pos = "POS_score"
    lev = c('negative', 'Serous-like')
    mm = "CNV-H"
    sw = TRUE
    clr = "#F6AE2D"
  } else{
    pos = "POS_score"
    lev = c('negative', f[2])
    mm = f[2]
    sw = TRUE
    clr = "#F26419"
  }
  i = paste(f[1], f[2], sep="")
  Test_tile <- read.csv(paste("~/documents/CPTAC-UCEC/Results/NL5/", i, "/out/Test_slide.csv", sep=''))
  answersa <- factor(Test_tile$True_label)
  resultsa <- factor(Test_tile$Prediction)
  roca <- plot(roc(answersa, Test_tile[[pos]], levels=lev), print.auc = FALSE, col = clr, add = sw, labels = FALSE, tck = -0.02)
}
legend("bottomright", legend=c("Histology", "CNV-H (Endometrioid)", "CNV-H", 'TP53'),
       col=c("#55DDE0", "#33658A", "#F6AE2D", "#F26419"), lwd=2)
dev.off()

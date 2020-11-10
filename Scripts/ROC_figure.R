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


# Figure 6 ROC
library(pROC)
todolist = list(c('X1histology_NYU_NL5', 'F3histology_NYU_NL6','his', 'X1', 'F3'), c('X3CNVH_NYU_NL5', 'X4CNVH_NYU_NL6', 'CNVH', 'X3', 'X4'), c('X1TP53_NYU_NL5', 'I1TP53_NYU_NL6','TP53', 'X1', 'I1'), c('I5MSI_NYU_NL5', 'X4MSI_NYU_NL6', 'MSIst', 'I5', 'X4'))
color = c("#F6AE2D", "#F26419", "#55DDE0", "#33658A")
for (f in todolist){
  if (f[3] == 'his'){
    pdf(file=paste("~/documents/CPTAC-UCEC/Manuscript/Figures/ROC/", f[3],"_ROC_slide.pdf", sep=''),
        width=6,height=6)
    pos = "Serous_score"
    lev = c('Endometrioid', 'Serous')
    mm = 'Histology'
    sw = TRUE
    i = paste(f[4], f[3], sep="")
    Test_tile <- read.csv(paste("~/documents/CPTAC-UCEC/Results/NL5/", i, "/out/Test_slide.csv", sep=''))
    answersa <- factor(Test_tile$True_label)
    resultsa <- factor(Test_tile$Prediction)
    roca <- plot(roc(answersa, Test_tile[[pos]], levels=lev), print.auc = FALSE, col = color[1], add = FALSE, labels = FALSE, tck = -0.02)
    
    Test_tile <- read.csv(paste("~/documents/CPTAC-UCEC/Results/NYU_test/", f[1], "/out/Test_slide.csv", sep=''))
    answersa <- factor(Test_tile$True_label)
    resultsa <- factor(Test_tile$Prediction)
    roca <- plot(roc(answersa, Test_tile[[pos]], levels=lev), print.auc = FALSE, col = color[2], add = sw, labels = FALSE, tck = -0.02)
    
    # i = paste(f[5], f[3], sep="")
    # Test_tile <- read.csv(paste("~/documents/CPTAC-UCEC/Results/NL6/", i, "/out/Test_slide.csv", sep=''))
    # answersa <- factor(Test_tile$True_label)
    # resultsa <- factor(Test_tile$Prediction)
    # roca <- plot(roc(answersa, Test_tile[[pos]], levels=lev), print.auc = FALSE, col = color[3], add = sw, labels = FALSE, tck = -0.02)
    # 
    # Test_tile <- read.csv(paste("~/documents/CPTAC-UCEC/Results/NYU_test/", f[2], "/out/Test_slide.csv", sep=''))
    # answersa <- factor(Test_tile$True_label)
    # resultsa <- factor(Test_tile$Prediction)
    # roca <- plot(roc(answersa, Test_tile[[pos]], levels=lev), print.auc = FALSE, col = color[4], add = sw, labels = FALSE, tck = -0.02)
    # legend("bottomright", legend=c("Mixed (0.969)", "Mixed on NYU set (0.913)", "Independent (0.962)", 'Independent on NYU set (0.827)'),
    #        col=c("#55DDE0", "#33658A", "#F6AE2D", "#F26419"), lwd=2)
    legend("bottomright", legend=c("Mixed test set (AUROC=0.969)", "NYU test set (AUROC=0.913)"),
           col=c("#F6AE2D", "#F26419"), lwd=2)
    
    dev.off()
    
  } else if(f[3] == 'MSIst'){
    pdf(file=paste("~/documents/CPTAC-UCEC/Manuscript/Figures/ROC/", f[3],"_ROC_slide.pdf", sep=''),
        width=6,height=6)
    pos = "MSI.H_score"
    lev = c('MSS', 'MSI-H')
    mm = 'MSI-high'
    sw = TRUE
    i = paste(f[4], f[3], sep="")
    Test_tile <- read.csv(paste("~/documents/CPTAC-UCEC/Results/NL5/", i, "/out/Test_slide.csv", sep=''))
    answersa <- factor(Test_tile$True_label)
    resultsa <- factor(Test_tile$Prediction)
    roca <- plot(roc(answersa, Test_tile[[pos]], levels=lev), print.auc = FALSE, col = color[1], add = FALSE, labels = FALSE, tck = -0.02)
    
    Test_tile <- read.csv(paste("~/documents/CPTAC-UCEC/Results/NYU_test/", f[1], "/out/Test_slide.csv", sep=''))
    answersa <- factor(Test_tile$True_label)
    resultsa <- factor(Test_tile$Prediction)
    roca <- plot(roc(answersa, Test_tile[["POS_score"]], levels=c('negative','MSI')), print.auc = FALSE, col = color[2], add = sw, labels = FALSE, tck = -0.02)
    
    # i = paste(f[5], f[3], sep="")
    # Test_tile <- read.csv(paste("~/documents/CPTAC-UCEC/Results/NL6/", i, "/out/Test_slide.csv", sep=''))
    # answersa <- factor(Test_tile$True_label)
    # resultsa <- factor(Test_tile$Prediction)
    # roca <- plot(roc(answersa, Test_tile[[pos]], levels=lev), print.auc = FALSE, col = color[3], add = sw, labels = FALSE, tck = -0.02)
    # 
    # Test_tile <- read.csv(paste("~/documents/CPTAC-UCEC/Results/NYU_test/", f[2], "/out/Test_slide.csv", sep=''))
    # answersa <- factor(Test_tile$True_label)
    # resultsa <- factor(Test_tile$Prediction)
    # roca <- plot(roc(answersa, Test_tile[["POS_score"]], levels=c('negative', 'MSI')), print.auc = FALSE, col = color[4], add = sw, labels = FALSE, tck = -0.02)
    # legend("bottomright", legend=c("Mixed (0.827)", "Mixed on NYU set (0.667)", "Independent (0.73)", 'Independent on NYU set (0.556)'),
    #        col=c("#55DDE0", "#33658A", "#F6AE2D", "#F26419"), lwd=2)
    legend("bottomright", legend=c("Mixed test set (AUROC=0.827)", "NYU test set (AUROC=0.667)"),
           col=c("#F6AE2D", "#F26419"), lwd=2)
    dev.off()
  } else if(f[3] == 'CNVH'){
    pdf(file=paste("~/documents/CPTAC-UCEC/Manuscript/Figures/ROC/", f[3],"_ROC_slide.pdf", sep=''),
        width=6,height=6)
    pos = "POS_score"
    lev = c('negative', 'Serous-like')
    mm = "CNV-H"
    sw = TRUE
    i = paste(f[4], f[3], sep="")
    Test_tile <- read.csv(paste("~/documents/CPTAC-UCEC/Results/NL5/", i, "/out/Test_slide.csv", sep=''))
    answersa <- factor(Test_tile$True_label)
    resultsa <- factor(Test_tile$Prediction)
    roca <- plot(roc(answersa, Test_tile[[pos]], levels=lev), print.auc = FALSE, col = color[1], add = FALSE, labels = FALSE, tck = -0.02)
    
    Test_tile <- read.csv(paste("~/documents/CPTAC-UCEC/Results/NYU_test/", f[1], "/out/Test_slide.csv", sep=''))
    answersa <- factor(Test_tile$True_label)
    resultsa <- factor(Test_tile$Prediction)
    roca <- plot(roc(answersa, Test_tile[[pos]], levels=lev), print.auc = FALSE, col = color[2], add = sw, labels = FALSE, tck = -0.02)
    
    # i = paste(f[5], f[3], sep="")
    # Test_tile <- read.csv(paste("~/documents/CPTAC-UCEC/Results/NL6/", i, "/out/Test_slide.csv", sep=''))
    # answersa <- factor(Test_tile$True_label)
    # resultsa <- factor(Test_tile$Prediction)
    # roca <- plot(roc(answersa, Test_tile[[pos]], levels=lev), print.auc = FALSE, col = color[3], add = sw, labels = FALSE, tck = -0.02)
    # 
    # Test_tile <- read.csv(paste("~/documents/CPTAC-UCEC/Results/NYU_test/", f[2], "/out/Test_slide.csv", sep=''))
    # answersa <- factor(Test_tile$True_label)
    # resultsa <- factor(Test_tile$Prediction)
    # roca <- plot(roc(answersa, Test_tile[[pos]], levels=lev), print.auc = FALSE, col = color[4], add = sw, labels = FALSE, tck = -0.02)
    # legend("bottomright", legend=c("Mixed (0.934)", "Mixed on NYU set (0.795)", "Independent (0.87)", 'Independent on NYU set (0.833)'),
    #        col=c("#55DDE0", "#33658A", "#F6AE2D", "#F26419"), lwd=2)
    legend("bottomright", legend=c("Mixed test set (AUROC=0.934)", "NYU test set (AUROC=0.795)"),
           col=c("#F6AE2D", "#F26419"), lwd=2)
    dev.off()
    
  } else{
    pdf(file=paste("~/documents/CPTAC-UCEC/Manuscript/Figures/ROC/", f[3],"_ROC_slide.pdf", sep=''),
        width=6,height=6)
    pos = "POS_score"
    lev = c('negative', f[3])
    mm = f[3]
    sw = TRUE
    i = paste(f[4], f[3], sep="")
    Test_tile <- read.csv(paste("~/documents/CPTAC-UCEC/Results/NL5/", i, "/out/Test_slide.csv", sep=''))
    answersa <- factor(Test_tile$True_label)
    resultsa <- factor(Test_tile$Prediction)
    roca <- plot(roc(answersa, Test_tile[[pos]], levels=lev), print.auc = FALSE, col = color[1], add = FALSE, labels = FALSE, tck = -0.02)
    
    Test_tile <- read.csv(paste("~/documents/CPTAC-UCEC/Results/NYU_test/", f[1], "/out/Test_slide.csv", sep=''))
    answersa <- factor(Test_tile$True_label)
    resultsa <- factor(Test_tile$Prediction)
    roca <- plot(roc(answersa, Test_tile[[pos]], levels=lev), print.auc = FALSE, col = color[2], add = sw, labels = FALSE, tck = -0.02)
    
    # i = paste(f[5], f[3], sep="")
    # Test_tile <- read.csv(paste("~/documents/CPTAC-UCEC/Results/NL6/", i, "/out/Test_slide.csv", sep=''))
    # answersa <- factor(Test_tile$True_label)
    # resultsa <- factor(Test_tile$Prediction)
    # roca <- plot(roc(answersa, Test_tile[[pos]], levels=lev), print.auc = FALSE, col = color[3], add = sw, labels = FALSE, tck = -0.02)
    # 
    # Test_tile <- read.csv(paste("~/documents/CPTAC-UCEC/Results/NYU_test/", f[2], "/out/Test_slide.csv", sep=''))
    # answersa <- factor(Test_tile$True_label)
    # resultsa <- factor(Test_tile$Prediction)
    # roca <- plot(roc(answersa, Test_tile[[pos]], levels=lev), print.auc = FALSE, col = color[4], add = sw, labels = FALSE, tck = -0.02)
    # legend("bottomright", legend=c("Mixed (0.873)", "Mixed on NYU set (0.92)", "Independent (0.767)", 'Independent on NYU set (0.941)'),
    #        col=c("#55DDE0", "#33658A", "#F6AE2D", "#F26419"), lwd=2)
    legend("bottomright", legend=c("Mixed test set (AUROC=0.873)", "NYU test set (AUROC=0.92)"),
           col=c("#F6AE2D", "#F26419"), lwd=2)
    dev.off()
  }
}




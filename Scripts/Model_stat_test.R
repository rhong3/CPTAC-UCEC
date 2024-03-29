# Statistical testing of different DL models and comparing them
library(readxl)
library(pROC)
subtype=read_excel('~/documents/CPTAC-UCEC/Results/Summary.xlsx', sheet = 1)
subtype=subtype[which(subtype$Tiles == "NL5"), ]
mutation=read_excel('~/documents/CPTAC-UCEC/Results/Summary.xlsx', sheet = 2)
mutation=mutation[which(mutation$Tiles == "NL5"), ]
histology=read_excel('~/documents/CPTAC-UCEC/Results/Summary.xlsx', sheet = 3)
histology=histology[which(histology$Tiles == "NL5"), ]
MSI=read_excel('~/documents/CPTAC-UCEC/Results/Summary.xlsx', sheet = 4)
MSI=MSI[which(MSI$Tiles == "NL5"), ]
special=read_excel('~/documents/CPTAC-UCEC/Results/Summary.xlsx', sheet = 5)
special=special[which(special$Tiles == "NL5"), ]

# For patient AUROC Wilcoxon
df = mutation[which(mutation$Architecture == "I1"), c(1,6)]
colnames(df) = c('Feature', "I1")
his = histology[which(histology$Architecture == "I1"), 5]
his$Feature = 'histology'
colnames(his) = c("I1", 'Feature')
df = rbind(df, his)
MM = MSI[which(MSI$Architecture == "I1"), 5]
MM$Feature = 'MSI'
colnames(MM) = c("I1", 'Feature')
df = rbind(df, MM)

arch = c('I2', 'I3', 'I4', 'I5', 'I6', 'X1', 'X2', 'X3', 'X4', 'F1', 'F2', 'F3', 'F4')
for (a in arch){
  temp = mutation[which(mutation$Architecture == a), c(1,6)]
  colnames(temp) = c('Feature', a)
  temp_his = histology[which(histology$Architecture == a), 5]
  temp_his$Feature = 'histology'
  colnames(temp_his) = c(a, 'Feature')
  temp_his = na.omit(temp_his)
  temp = rbind(temp, temp_his)
  temp_MM = MSI[which(MSI$Architecture == a), 5]
  temp_MM$Feature = 'MSI'
  colnames(temp_MM) = c(a, 'Feature')
  temp_MM = na.omit(temp_MM)
  temp = rbind(temp, temp_MM)
  df = merge(df, temp, by="Feature")
}

df = df[c(6,8,9,12,17,19,20),]


wilcoxon = data.frame(matrix('', ncol=14, nrow=14),stringsAsFactors = FALSE)
colnames(wilcoxon) = c('I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'X1', 'X2', 'X3', 'X4', 'F1', 'F2', 'F3', 'F4')
rownames(wilcoxon) = c('I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'X1', 'X2', 'X3', 'X4', 'F1', 'F2', 'F3', 'F4')
for (m in 2:15){
  for (n in 2:15){
    if (m<n){
      res <- wilcox.test(df[,c(m)], df[,c(n)], paired = TRUE)
      wilcoxon[m-1, n-1] = round(res$p.value, digits = 5)
      if (res$p.value <= 0.05){
        print(c(colnames(df)[c(m,n)],round(res$p.value, digits = 5), "!Significant!"))
      }
    }
  }
}

write.csv(wilcoxon, file = "~/documents/CPTAC-UCEC/Results/AUROC_test/Wilcoxon_patient_AUROC.csv", row.names=TRUE)


# ROC test
arch = c('I5', 'I6', 'X2', 'X1', 'X4', 'X3', 'F2', 'F1', 'F4', 'F3')
features = c('his', 'MSIst', 'FAT1', 'TP53', 'PTEN', 'ZFHX3', 'ARID1A', 'ATM', 'BRCA2', 'CTCF', 'CTNNB1', 'FBXW7', 'JAK1', 'KRAS', 'MTOR', 'PIK3CA', 'PIK3R1', 'PPP2R1A', 'RPL22', 'FGFR2')

for (f in features){
  PA_test = data.frame(matrix('', ncol=10, nrow=10),stringsAsFactors = FALSE)
  colnames(PA_test) = c('I5', 'I6', 'X2', 'X1', 'X4', 'X3', 'F2', 'F1', 'F4', 'F3')
  rownames(PA_test) = c('I5', 'I6', 'X2', 'X1', 'X4', 'X3', 'F2', 'F1', 'F4', 'F3')
  
  TI_test = data.frame(matrix('', ncol=10, nrow=10),stringsAsFactors = FALSE)
  colnames(TI_test) = c('I5', 'I6', 'X2', 'X1', 'X4', 'X3', 'F2', 'F1', 'F4', 'F3')
  rownames(TI_test) = c('I5', 'I6', 'X2', 'X1', 'X4', 'X3', 'F2', 'F1', 'F4', 'F3')
  if (f == 'his'){
    pos = "Serous_score"
    lev = c('Endometrioid', 'Serous')  
  } else if(f == 'MSIst'){
    pos = "MSI.H_score"
    lev = c('MSS', 'MSI-H')
  } else{
    pos = "POS_score"
    lev = c('negative', f)
  }
  for (ara in 1:10){
    for (arb in 1:10){
      if ((ara<arb) & (ara %% 2 == arb %% 2)){
        i = paste(arch[ara], f, sep='')
        j = paste(arch[arb], f, sep='')
        
        Test_slidea <- read.csv(paste("~/documents/CPTAC-UCEC/Results/NL5/", i, "/out/Test_slide.csv", sep=''))
        Test_slideb <- read.csv(paste("~/documents/CPTAC-UCEC/Results/NL5/", j, "/out/Test_slide.csv", sep=''))
        # per patient level
        answersa <- factor(Test_slidea$True_label)
        resultsa <- factor(Test_slidea$Prediction)
        answersb <- factor(Test_slideb$True_label)
        resultsb <- factor(Test_slideb$Prediction)
        # ROC
        roca =  roc(answersa, Test_slidea[[pos]], levels=lev)
        rocb =  roc(answersb, Test_slideb[[pos]], levels=lev)
        testa = roc.test(roca, rocb, method="delong", alternative="less")
        
        PA_test[ara, arb] = round(testa$p.value, digits = 5)
        
        Test_tilea <- read.csv(paste("~/documents/CPTAC-UCEC/Results/NL5/", i, "/out/Test_tile.csv", sep=''))
        Test_tileb <- read.csv(paste("~/documents/CPTAC-UCEC/Results/NL5/", j, "/out/Test_tile.csv", sep=''))
        # per patient level
        answersa <- factor(Test_tilea$True_label)
        resultsa <- factor(Test_tilea$Prediction)
        answersb <- factor(Test_tileb$True_label)
        resultsb <- factor(Test_tileb$Prediction)
        # ROC
        roca =  roc(answersa, Test_tilea[[pos]], levels=lev)
        rocb =  roc(answersb, Test_tileb[[pos]], levels=lev)
        testb = roc.test(roca, rocb, method="delong", alternative="greater")
        
        TI_test[ara, arb] = round(testb$p.value, digits = 5)

      }
    }
  }
  write.csv(PA_test, file = paste("~/documents/CPTAC-UCEC/Results/AUROC_test/less_",f ,"_patient_AUROC_test.csv", sep=''), row.names=TRUE)
  write.csv(TI_test, file = paste("~/documents/CPTAC-UCEC/Results/AUROC_test/less_",f ,"_tile_AUROC_test.csv", sep=''), row.names=TRUE)
}


# Bootstrap t-test
# ROC test
arch = c('I5', 'I6', 'X2', 'X1', 'X4', 'X3', 'F2', 'F1', 'F4', 'F3')
features = c('his', 'MSIst', 'FAT1', 'TP53', 'PTEN', 'ZFHX3', 'ARID1A', 'ATM', 'BRCA2', 'CTCF', 'CTNNB1', 'FBXW7', 'JAK1', 'KRAS', 'MTOR', 'PIK3CA', 'PIK3R1', 'PPP2R1A', 'RPL22', 'FGFR2', 'SL', 'CNVH')

for (f in features){
  PA_test = data.frame(matrix('', ncol=10, nrow=10),stringsAsFactors = FALSE)
  colnames(PA_test) = c('I5', 'I6', 'X2', 'X1', 'X4', 'X3', 'F2', 'F1', 'F4', 'F3')
  rownames(PA_test) = c('I5', 'I6', 'X2', 'X1', 'X4', 'X3', 'F2', 'F1', 'F4', 'F3')
  
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
  for (ara in 1:10){
    for (arb in 1:10){
      if ((ara<arb) & (ara %% 2 == arb %% 2)){
        i = paste(arch[ara], f, sep='')
        j = paste(arch[arb], f, sep='')
        
        Test_slidea <- read.csv(paste("~/documents/CPTAC-UCEC/Results/NL5/", i, "/out/Test_slide.csv", sep=''))
        Test_slideb <- read.csv(paste("~/documents/CPTAC-UCEC/Results/NL5/", j, "/out/Test_slide.csv", sep=''))

        Sprla = list()
        Sprlb = list()
        for (j in 1:50){
          sampleddfa = Test_slidea[sample(nrow(Test_slidea), round(nrow(Test_slidea)*0.9)),]
          answersa <- factor(sampleddfa$True_label)
          resultsa <- factor(sampleddfa$Prediction)
          roca =  roc(answersa, sampleddfa[[pos]], levels=lev)
          Sprla[j] = roca$auc
          
          sampleddfb = Test_slideb[sample(nrow(Test_slideb), round(nrow(Test_slideb)*0.9)),]
          answersb <- factor(sampleddfb$True_label)
          resultsb <- factor(sampleddfb$Prediction)
          rocb =  roc(answersb, sampleddfb[[pos]], levels=lev)
          Sprlb[j] = rocb$auc
        }
        
        PA_test[ara, arb] = t.test(as.numeric(Sprla), as.numeric(Sprlb), alternative = "less", paired = FALSE)$p.value
        
      }
    }
  }
  write.csv(PA_test, file = paste("~/documents/CPTAC-UCEC/Results/t-test/less_",f ,"_patient_AUROC_test.csv", sep=''), row.names=TRUE)
} 


# t-test box plot 
library(ggplot2)
library(ggpubr)
features = c('his', 'MSIst', 'FAT1', 'TP53', 'PTEN', 'ZFHX3', 'ARID1A', 'ATM', 'BRCA2', 'CTCF', 'CTNNB1', 'FBXW7', 'JAK1', 'KRAS', 'MTOR', 'PIK3CA', 'PIK3R1', 'PPP2R1A', 'RPL22', 'FGFR2', 'SL', 'CNVH')
compa = list(c('I5', 'X2'), c('X2', 'X4'), c('X2', 'F2'), c('X4', 'F4'), c('I6', 'X1'), c('X1', 'X3'), c('X1', 'F1'), c('X3', 'F3'))
for (f in features){
  all = data.frame(Slide_AUC= numeric(0), Tile_AUC= numeric(0), Architecture= character(0))
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
  arch = c('I5', 'I6', 'X2', 'X1', 'X4', 'X3', 'F2', 'F1', 'F4', 'F3')
  for (a in arch){
    i = paste(a, f, sep='')
    Test_slide <- read.csv(paste("~/documents/CPTAC-UCEC/Results/NL5/", i, "/out/Test_slide.csv", sep=''))
    Test_tile <- read.csv(paste("~/documents/CPTAC-UCEC/Results/NL5/", i, "/out/Test_tile.csv", sep=''))
    Sprla = list()
    Sprlb = list()
    for (j in 1:50){
      sampleddfa = Test_slide[sample(nrow(Test_slide), round(nrow(Test_slide)*0.8)),]
      answersa <- factor(sampleddfa$True_label)
      resultsa <- factor(sampleddfa$Prediction)
      roca =  roc(answersa, sampleddfa[[pos]], levels=lev)
      Sprla[j] = roca$auc
      
      sampleddfb = Test_tile[sample(nrow(Test_tile), round(nrow(Test_tile)*0.8)),]
      answersb <- factor(sampleddfb$True_label)
      resultsb <- factor(sampleddfb$Prediction)
      rocb =  roc(answersb, sampleddfb[[pos]], levels=lev)
      Sprlb[j] = rocb$auc
    }
    temp_all= data.frame(Slide_AUC=as.numeric(Sprla), Tile_AUC=as.numeric(Sprlb), Architecture=a)
    all = rbind(all, temp_all)
  }
  pp = ggboxplot(all, x = "Architecture", y = "Slide_AUC",
                 color = "black", fill = "Architecture", palette = "grey")+ 
    stat_compare_means(method = "t.test", method.args = list(alternative = "less"), comparisons = compa, label = "p.signif")
  pl = ggboxplot(all, x = "Architecture", y = "Tile_AUC",
                 color = "black", fill = "Architecture", palette = "grey")+ 
    stat_compare_means(method = "t.test", method.args = list(alternative = "less"), comparisons = compa, label = "p.signif") 
  
  pdf(file=paste("~/documents/CPTAC-UCEC/Results/t-test/", f, ".pdf", sep=''),
      width=28,height=7)
  grid.arrange(pp,pl,nrow=1, ncol=2)
  dev.off()
}


# t-test task based
library(ggplot2)
library(ggpubr)
features = c('SL', 'CNVH', 'CNVL')
arch = c("I5", 'X2', 'X4', 'F2', 'F4', 'I6', 'X1', 'X3', 'F1', 'F3')

all = data.frame(Slide_AUC= numeric(0), Tile_AUC= numeric(0), Architecture= character(0), Feature=character(0))
for (a in arch){
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
    } else if(f == 'CNVL'){
      pos = "POS_score"
      lev = c('negative', 'Endometrioid') 
    } else{
      pos = "POS_score"
      lev = c('negative', f)
    }
    i = paste(a, f, sep="")
    i = paste(a, f, sep='')
    Test_slide <- read.csv(paste("~/documents/CPTAC-UCEC/Results/NL5/", i, "/out/Test_slide.csv", sep=''))
    Test_tile <- read.csv(paste("~/documents/CPTAC-UCEC/Results/NL5/", i, "/out/Test_tile.csv", sep=''))
    Sprla = list()
    Sprlb = list()
    for (j in 1:50){
      sampleddfa = Test_slide[sample(nrow(Test_slide), round(nrow(Test_slide)*0.8)),]
      answersa <- factor(sampleddfa$True_label)
      resultsa <- factor(sampleddfa$Prediction)
      roca =  roc(answersa, sampleddfa[[pos]], levels=lev)
      Sprla[j] = roca$auc
      
      sampleddfb = Test_tile[sample(nrow(Test_tile), round(nrow(Test_tile)*0.8)),]
      answersb <- factor(sampleddfb$True_label)
      resultsb <- factor(sampleddfb$Prediction)
      rocb =  roc(answersb, sampleddfb[[pos]], levels=lev)
      Sprlb[j] = rocb$auc
    }
    temp_all= data.frame(Patient_AUC=as.numeric(Sprla), Tile_AUC=as.numeric(Sprlb), Architecture=a, Feature=f)
    all = rbind(all, temp_all)
  }
}

colnames(all) = c("Patient_AUC", "Tile_AUC", "Architecture", "Feature")
write.csv(all, file = "~/documents/CPTAC-UCEC/Results/t-test/bootstrap_80%_50.csv", row.names=FALSE)

# For figure
library(ggplot2)
library(ggpubr)
all = read.csv("~/documents/CPTAC-UCEC/Results/t-test/bootstrap_80%_50.csv")
all$Feature <- gsub('his', 'Histology', all$Feature)
all$Feature <- gsub('SL', 'CNV-H (Endometrioid)', all$Feature)
all$Feature <- gsub('CNVH', 'CNV-H', all$Feature)
all$Feature <- gsub('CNVL', 'CNV-L', all$Feature)
all$Feature <- gsub('MSIst', 'MSI-high', all$Feature)

pair = list(c('I6', 'X1'), c('I5', 'X2'), c('X1', 'X3'), c('X2', 'X4'), c('X1', 'F1'), c('X2', 'F2'), c('X3', 'F3'), c('X4', 'F4'))

for (pwa in pair){
  wa = pwa[1]
  wb = pwa[2]
  all_sub = all[all["Architecture"] == wa | all["Architecture"] == wb, ]
  all_sub = all_sub[all_sub$Feature %in% c('Histology', 'MSI-high', 'FAT1', 'TP53', 'PTEN', 'ZFHX3', 'CNV-H (Endometrioid)', 'CNV-H', "CNV-L"), ]
  
  pp = ggboxplot(all_sub, x = "Feature", y = "Patient_AUC",
                 color = "black", fill = "Architecture", palette = "grey")+ 
    stat_compare_means(method = "t.test", method.args = list(alternative = "greater"), aes(group = Architecture), label = "p.signif", label.y = 1.05)+
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  pl = ggboxplot(all_sub, x = "Feature", y = "Tile_AUC",
                 color = "black", fill = "Architecture", palette = "grey")+ 
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  pdf(file=paste("~/documents/CPTAC-UCEC/Results/t-test/", wa, wb, "_lite.pdf", sep=''),
      width=20,height=5)
  grid.arrange(pp,pl,nrow=1, ncol=2)
  dev.off()
} 


# For figure barplots
data_summary <- function(data, varname, groupnames){
  require(plyr)
  summary_func <- function(x, col){
    c(mean = mean(x[[col]], na.rm=TRUE),
      sd = sd(x[[col]], na.rm=TRUE))
  }
  data_sum<-ddply(data, groupnames, .fun=summary_func,
                  varname)
  data_sum <- rename(data_sum, c("mean" = varname))
  return(data_sum)
}

library(ggplot2)
library(ggpubr)
library(gridExtra)

all = read.csv("~/documents/CPTAC-UCEC/Results/t-test/bootstrap_80%_50.csv")
all$Feature <- gsub('his', 'Histology', all$Feature)
all$Feature <- gsub('SL', 'CNV-H (Endometrioid)', all$Feature)
all$Feature <- gsub('CNVH', 'CNV-H', all$Feature)
all$Feature <- gsub('CNVL', 'CNV-L', all$Feature)
all$Feature <- gsub('MSIst', 'MSI-high', all$Feature)
all$Architecture = gsub("F1", "Z1", all$Architecture)
all$Architecture = gsub("F2", "Z2", all$Architecture)
all$Architecture = gsub("F3", "Z3", all$Architecture)
all$Architecture = gsub("F4", "Z4", all$Architecture)

pair = list(c('I6', 'X1'), c('I5', 'X2'), c('X1', 'X3'), c('X2', 'X4'), c('X1', 'Z1'), c('X2', 'Z2'), c('X3', 'Z3'), c('X4', 'Z4'))

for (pwa in pair){
  wa = pwa[1]
  wb = pwa[2]
  all_sub = all[all["Architecture"] == wa | all["Architecture"] == wb, ]
  all_sub = all_sub[all_sub$Feature %in% c('Histology', 'MSI-high', 'FAT1', 'TP53', 'PTEN', 'ZFHX3', 'CNV-H (Endometrioid)', 'CNV-H', "CNV-L"), ]
  all_sub.x = data_summary(all_sub, varname="Patient_AUC", 
                         groupnames=c("Architecture", "Feature"))
  all_sub.y = data_summary(all_sub, varname="Tile_AUC", 
                           groupnames=c("Architecture", "Feature"))
  
  pp<- ggplot(all_sub.x, aes(x=reorder(Feature, -Patient_AUC), y=Patient_AUC, fill=Architecture)) + 
    geom_bar(stat="identity", color="black",
             position=position_dodge()) +
    scale_fill_manual(values=c("#D3D3D3", "#808080")) +
    geom_errorbar(aes(ymin=Patient_AUC-sd, ymax=Patient_AUC+sd), width=.2,
                  position=position_dodge(.9)) +
    stat_compare_means(data = all_sub, method = "t.test", method.args = list(alternative = "greater"), aes(group = Architecture), label = "p.signif", label.y = 1.1, size=8)+
    theme(axis.text.x = element_text(angle = 30, hjust = 1), panel.border = element_blank(), panel.grid.major = element_blank(), panel.background = element_blank(),
          panel.grid.minor = element_blank(), legend.position = "none")+  
    font("xlab", size = 0, color = "black")+
    font("ylab", size = 15, color = "black", face = "bold")+
    font("xy.text", size = 15, color = "black", face = "bold")
  
  
  pl<- ggplot(all_sub.y, aes(x=reorder(Feature, -Tile_AUC), y=Tile_AUC, fill=Architecture)) + 
    geom_bar(stat="identity", color="black",
             position=position_dodge()) +
    scale_fill_manual(values=c("#D3D3D3", "#808080")) +
    geom_errorbar(aes(ymin=Tile_AUC-sd, ymax=Tile_AUC+sd), width=.2,
                  position=position_dodge(.9)) +
    stat_compare_means(data = all_sub, method = "t.test", method.args = list(alternative = "greater"), aes(group = Architecture), label = "p.signif", label.y = 1.1, size=8)+
    theme(axis.text.x = element_text(angle = 30, hjust = 1), panel.border = element_blank(), panel.grid.major = element_blank(),panel.background = element_blank(),
          panel.grid.minor = element_blank(), legend.position = "none")+  
    font("xlab", size = 0, color = "black")+
    font("ylab", size = 15, color = "black", face = "bold")+
    font("xy.text", size = 15, color = "black", face = "bold")
  
  pdf(file=paste("~/documents/CPTAC-UCEC/Results/t-test/", wa, wb, "_bar_lite.pdf", sep=''),
      width=20,height=5)
  grid.arrange(pp,pl,nrow=1, ncol=2)
  dev.off()
} 


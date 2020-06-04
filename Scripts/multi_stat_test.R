library(readxl)
library(pROC)
library(ggplot2)
library(ggpubr)

# Multi-level vs. Panoptes tests
# t-test task based sampling
library(ggplot2)
library(ggpubr)
features = c('SL', 'CNVH', 'his', 'MSIst', 'FAT1', 'TP53', 'PTEN', 'ZFHX3', 'ARID1A', 'ATM', 'BRCA2', 'CTCF', 'CTNNB1', 'FBXW7', 'JAK1', 'KRAS', 'MTOR', 'PIK3CA', 'PIK3R1', 'PPP2R1A', 'RPL22', 'FGFR2')
arch = c("I5", 'I6')
oldsampled = read.csv("~/documents/CPTAC-UCEC/Results/t-test/bootstrap_80%_50.csv")
oldsampled$Tile = 'NL5'

all = data.frame(Slide_AUC= numeric(0), Tile_AUC= numeric(0), Architecture= character(0), Feature=character(0))
for (a in arch){
  for (nl in c('NL8', 'NL9')){
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
      i = paste(a, f, sep='')
      i = paste(a, f, sep='')
      Test_slide <- read.csv(paste("~/documents/CPTAC-UCEC/Results/", nl, "/", i, "/out/Test_slide.csv", sep=''))
      Test_tile <- read.csv(paste("~/documents/CPTAC-UCEC/Results/", nl, "/", i, "/out/Test_tile.csv", sep=''))

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
      temp_all= data.frame(Slide_AUC=as.numeric(Sprla), Tile_AUC=as.numeric(Sprlb), Architecture=a, Feature=f, Tile=nl)
      all = rbind(all, temp_all)
    }
  }
}

colnames(all) = c("Patient_AUC", "Tile_AUC", "Architecture", "Feature", "Tile")
all = rbind(oldsampled, all)
write.csv(all, file = "~/documents/CPTAC-UCEC/Results/t-test-multi/bootstrap_80%_50.csv", row.names=FALSE)


# For figure
library(ggplot2)
library(ggpubr)
library(gridExtra)
all = read.csv("~/documents/CPTAC-UCEC/Results/t-test-multi/bootstrap_80%_50.csv")
all$Feature <- gsub('his', 'Histology', all$Feature)
all$Feature <- gsub('SL', 'CNV-H (Endometrioid)', all$Feature)
all$Feature <- gsub('CNVH', 'CNV-H', all$Feature)
all$Feature <- gsub('MSIst', 'MSI-high', all$Feature)

pair = list(c('I6', 'X1'), c('I5', 'X2'))

for (pwa in pair){
  wa = pwa[1]
  wb = pwa[2]
  all_sub = all[all["Architecture"] == wa | all["Architecture"] == wb, ]
  all_sub = all_sub[all_sub$Feature %in% c('Histology', 'MSI-high', 'FAT1', 'TP53', 'PTEN', 'ZFHX3', 'CNV-H (Endometrioid)', 'CNV-H'), ]
  
  pp = ggboxplot(all_sub, x = "Feature", y = "Patient_AUC",
                 color = "black", fill = "Architecture", palette = "grey")+ 
    stat_compare_means(method = "t.test", method.args = list(alternative = "greater"), aes(group = Architecture), label = "p.signif", label.y = 1.05)+
    stat_compare_means(method = "t.test", method.args = list(alternative = "greater"), aes(group = Architecture), label = "p.format", label.y = 1.1)+
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  pl = ggboxplot(all_sub, x = "Feature", y = "Tile_AUC",
                 color = "black", fill = "Architecture", palette = "grey")+ 
    stat_compare_means(method = "t.test", method.args = list(alternative = "greater"), aes(group = Architecture), label = "p.signif", label.y = 1.05)+
    stat_compare_means(method = "t.test", method.args = list(alternative = "greater"), aes(group = Architecture), label = "p.format", label.y = 1.1)+
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  pdf(file=paste("~/documents/CPTAC-UCEC/Results/t-test-multi/", wa, wb, "_lite.pdf", sep=''),
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

all = read.csv("~/documents/CPTAC-UCEC/Results/t-test-multi/bootstrap_80%_50.csv")
all$Feature <- gsub('his', 'Histology', all$Feature)
all$Feature <- gsub('SL', 'CNV-H (Endometrioid)', all$Feature)
all$Feature <- gsub('CNVH', 'CNV-H', all$Feature)
all$Feature <- gsub('MSIst', 'MSI-high', all$Feature)
all$Architecture = gsub("F1", "Z1", all$Architecture)
all$Architecture = gsub("F2", "Z2", all$Architecture)
all$Architecture = gsub("F3", "Z3", all$Architecture)
all$Architecture = gsub("F4", "Z4", all$Architecture)

pair = list(c('I6', 'X1'), c('I5', 'X2'))

for (pwa in pair){
  wa = pwa[1]
  wb = pwa[2]
  all_sub = all[all["Architecture"] == wa | all["Architecture"] == wb, ]
  all_sub = all_sub[all_sub$Feature %in% c('Histology', 'MSI-high', 'FAT1', 'TP53', 'PTEN', 'ZFHX3', 'CNV-H (Endometrioid)', 'CNV-H'), ]
  all_sub.x = data_summary(all_sub, varname="Patient_AUC", 
                           groupnames=c("Architecture", "Feature"))
  all_sub.y = data_summary(all_sub, varname="Tile_AUC", 
                           groupnames=c("Architecture", "Feature"))
  
  pp<- ggplot(all_sub.x, aes(x=Feature, y=Patient_AUC, fill=Architecture)) + 
    geom_bar(stat="identity", color="black",
             position=position_dodge()) +
    scale_fill_manual(values=c("#D3D3D3", "#808080")) +
    geom_errorbar(aes(ymin=Patient_AUC-sd, ymax=Patient_AUC+sd), width=.2,
                  position=position_dodge(.9)) +
    stat_compare_means(data = all_sub, method = "t.test", method.args = list(alternative = "greater"), aes(group = Architecture), label = "p.signif", label.y = 1.05)+
    stat_compare_means(data = all_sub, method = "t.test", method.args = list(alternative = "greater"), aes(group = Architecture), label = "p.format", label.y = 1.15)+theme_bw()+ 
    theme(axis.text.x = element_text(angle = 45, hjust = 1), panel.border = element_blank(), panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(), legend.position = "top")
  
  
  pl<- ggplot(all_sub.y, aes(x=Feature, y=Tile_AUC, fill=Architecture)) + 
    geom_bar(stat="identity", color="black",
             position=position_dodge()) +
    scale_fill_manual(values=c("#D3D3D3", "#808080")) +
    geom_errorbar(aes(ymin=Tile_AUC-sd, ymax=Tile_AUC+sd), width=.2,
                  position=position_dodge(.9)) +
    stat_compare_means(data = all_sub, method = "t.test", method.args = list(alternative = "greater"), aes(group = Architecture), label = "p.signif", label.y = 1.05)+
    stat_compare_means(data = all_sub, method = "t.test", method.args = list(alternative = "greater"), aes(group = Architecture), label = "p.format", label.y = 1.15)+theme_bw()+ 
    theme(axis.text.x = element_text(angle = 45, hjust = 1), panel.border = element_blank(), panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(), legend.position = "top")
  
  pdf(file=paste("~/documents/CPTAC-UCEC/Results/t-test-multi/", wa, wb, "_bar_lite.pdf", sep=''),
      width=20,height=5)
  grid.arrange(pp,pl,nrow=1, ncol=2)
  dev.off()
} 



#Single 2.5X and 5X with panoptes
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

all = read.csv("~/documents/CPTAC-UCEC/Results/t-test-multi/bootstrap_80%_50.csv")
all$Feature <- gsub('his', 'Histology', all$Feature)
all$Feature <- gsub('SL', 'CNV-H (Endometrioid)', all$Feature)
all$Feature <- gsub('CNVH', 'CNV-H', all$Feature)
all$Feature <- gsub('MSIst', 'MSI-high', all$Feature)
all$Architecture = gsub("F1", "Z1", all$Architecture)
all$Architecture = gsub("F2", "Z2", all$Architecture)
all$Architecture = gsub("F3", "Z3", all$Architecture)
all$Architecture = gsub("F4", "Z4", all$Architecture)

pair = list(c('I6', 'X1'), c('I5', 'X2'))

for (pwa in pair){
  wa = pwa[1]
  wb = pwa[2]
  all_sub = all[all["Architecture"] == wa | all["Architecture"] == wb, ]
  all_sub = all_sub[all_sub$Feature %in% c('Histology', 'MSI-high', 'FAT1', 'TP53', 'PTEN', 'ZFHX3', 'CNV-H (Endometrioid)', 'CNV-H'), ]
  
  all_sub89 = all_sub[all_sub["Tile"] == 'NL9' & all_sub["Architecture"] == wa,]
  all_sub5 = all_sub[all_sub["Tile"] == 'NL5' & all_sub["Architecture"] == wb,]
  all_sub = rbind(all_sub89, all_sub5)
  
  all_sub.x = data_summary(all_sub, varname="Patient_AUC", 
                           groupnames=c("Architecture", "Feature"))
  all_sub.y = data_summary(all_sub, varname="Tile_AUC", 
                           groupnames=c("Architecture", "Feature"))
  
  pp<- ggplot(all_sub.x, aes(x=Feature, y=Patient_AUC, fill=Architecture)) + 
    geom_bar(stat="identity", color="black",
             position=position_dodge()) +
    scale_fill_manual(values=c("#D3D3D3", "#808080")) +
    geom_errorbar(aes(ymin=Patient_AUC-sd, ymax=Patient_AUC+sd), width=.2,
                  position=position_dodge(.9)) +
    stat_compare_means(data = all_sub, method = "t.test", method.args = list(alternative = "greater"), aes(group = Architecture), label = "p.signif", label.y = 1.05)+
    stat_compare_means(data = all_sub, method = "t.test", method.args = list(alternative = "greater"), aes(group = Architecture), label = "p.format", label.y = 1.15)+theme_bw()+ 
    theme(axis.text.x = element_text(angle = 45, hjust = 1), panel.border = element_blank(), panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(), legend.position = "top")
  
  
  pl<- ggplot(all_sub.y, aes(x=Feature, y=Tile_AUC, fill=Architecture)) + 
    geom_bar(stat="identity", color="black",
             position=position_dodge()) +
    scale_fill_manual(values=c("#D3D3D3", "#808080")) +
    geom_errorbar(aes(ymin=Tile_AUC-sd, ymax=Tile_AUC+sd), width=.2,
                  position=position_dodge(.9)) +
    stat_compare_means(data = all_sub, method = "t.test", method.args = list(alternative = "greater"), aes(group = Architecture), label = "p.signif", label.y = 1.05)+
    stat_compare_means(data = all_sub, method = "t.test", method.args = list(alternative = "greater"), aes(group = Architecture), label = "p.format", label.y = 1.15)+theme_bw()+ 
    theme(axis.text.x = element_text(angle = 45, hjust = 1), panel.border = element_blank(), panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(), legend.position = "top")
  
  pdf(file=paste("~/documents/CPTAC-UCEC/Results/t-test-multi/NL9/", wa, wb, "_bar_lite.pdf", sep=''),
      width=20,height=5)
  grid.arrange(pp,pl,nrow=1, ncol=2)
  dev.off()
} 


# Compare different resolution
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

for (ar in c("I6", 'I5')){
  all = read.csv("~/documents/CPTAC-UCEC/Results/t-test-multi/bootstrap_80%_50.csv")
  all$Feature <- gsub('his', 'Histology', all$Feature)
  all$Feature <- gsub('SL', 'CNV-H (Endometrioid)', all$Feature)
  all$Feature <- gsub('CNVH', 'CNV-H', all$Feature)
  all$Feature <- gsub('MSIst', 'MSI-high', all$Feature)
  all$Architecture = gsub("F1", "Z1", all$Architecture)
  all$Architecture = gsub("F2", "Z2", all$Architecture)
  all$Architecture = gsub("F3", "Z3", all$Architecture)
  all$Architecture = gsub("F4", "Z4", all$Architecture)
  all$Tile <- gsub('NL5', '10X', all$Tile)
  all$Tile <- gsub('NL8', '5X', all$Tile)
  all$Tile <- gsub('NL9', '2.5X', all$Tile)
  all.1 = all[all['Tile'] == '2.5X', ]
  all.2 = all[all['Tile'] == '5X', ]
  all.3 = all[all['Tile'] == '10X', ]
  all = rbind(all.1, all.2, all.3)
  rownames(all) <- NULL
  all_sub = all[all['Architecture'] == ar, ]
  all_sub = all_sub[all_sub$Feature %in% c('Histology', 'MSI-high', 'FAT1', 'TP53', 'PTEN', 'ZFHX3', 'CNV-H (Endometrioid)', 'CNV-H'), ]

  all_sub.x = data_summary(all_sub, varname="Patient_AUC", 
                           groupnames=c("Tile", "Feature"))
  all_sub.y = data_summary(all_sub, varname="Tile_AUC", 
                           groupnames=c("Tile", "Feature"))
  
  pp<- ggplot(all_sub.x, aes(x=Feature, y=Patient_AUC, fill=Tile, group=Tile)) + 
    geom_bar(stat="identity", color="black",
             position=position_dodge()) +
    scale_fill_manual(values=c("#D3D3D3", "#808080", "#2A2A2A")) +
    geom_errorbar(aes(ymin=Patient_AUC-sd, ymax=Patient_AUC+sd), width=.2,
                  position=position_dodge(.9)) + theme_bw()+ 
    theme(axis.text.x = element_text(angle = 45, hjust = 1), panel.border = element_blank(), panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(), legend.position = "top")
  
  
  pl<- ggplot(all_sub.y, aes(x=Feature, y=Tile_AUC, fill=Tile, group=Tile)) + 
    geom_bar(stat="identity", color="black",
             position=position_dodge()) +
    scale_fill_manual(values=c("#D3D3D3", "#808080", "#2A2A2A")) +
    geom_errorbar(aes(ymin=Tile_AUC-sd, ymax=Tile_AUC+sd), width=.2,
                  position=position_dodge(.9)) + theme_bw()+ 
    theme(axis.text.x = element_text(angle = 45, hjust = 1), panel.border = element_blank(), panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(), legend.position = "top")
  
  pdf(file=paste("~/documents/CPTAC-UCEC/Results/t-test-multi/NLcompare/", ar, "_bar_lite.pdf", sep=''),
      width=20,height=5)
  grid.arrange(pp,pl,nrow=1, ncol=2)
  dev.off()
 
}



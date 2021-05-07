library(readxl)
library(pROC)
library(ggplot2)
library(ggpubr)
library(icesTAF)

# # Create joint decision models
# features = c('SL', 'CNVH', 'his', 'MSIst', 'FAT1', 'TP53', 'PTEN', 'ZFHX3', 'ARID1A', 'ATM', 'BRCA2', 'CTCF', 'CTNNB1', 'FBXW7', 'JAK1', 'KRAS', 'MTOR', 'PIK3CA', 'PIK3R1', 'PPP2R1A', 'RPL22', 'FGFR2')
# arch = c("I1", "I2", "I3", "I5", 'I6')
# 
# for (f in features){
#   if (f == 'his'){
#     pos = "Serous"
#     pos_score = "Serous_score"
#     neg = "Endometrioid"
#     neg_score = "Endometrioid_score"
#   } else if(f == 'MSIst'){
#     pos = "MSI-H"
#     pos_score = "MSI.H_score"
#     neg = "MSS"
#     neg_score = "MSS_score"
#   } else if(f == 'SL' | f == 'CNVH'){
#     pos = 'Serous-like'
#     pos_score = "POS_score"
#     neg = 'negative'
#     neg_score = "NEG_score"
#   } else{
#     pos = f
#     pos_score = "POS_score"
#     neg = 'negative'
#     neg_score = "NEG_score"
#   }
#   for (ar in arch){
#     mkdir(paste("~/documents/CPTAC-UCEC/Results/NLX/", ar, f, "/out/", sep=''))
#     NL5 = read.csv(paste("~/documents/CPTAC-UCEC/Results/NL5/", ar, f,"/out/Test_slide.csv", sep=''))[,-c(2,5)]
#     NL8 = read.csv(paste("~/documents/CPTAC-UCEC/Results/NL8/", ar, f,"/out/Test_slide.csv", sep=''))[,-c(2,5)]
#     NL9 = read.csv(paste("~/documents/CPTAC-UCEC/Results/NL9/", ar, f,"/out/Test_slide.csv", sep=''))[,-c(2,5)]
#     NLX = merge(merge(NL5,NL8, by = c('slide', 'True_label')), NL9, by = c('slide', 'True_label'))
#     NLX = na.omit(NLX)
#     NLX[pos_score] = round((NLX[,3]+NLX[,4]+NLX[,5])/3, 8)
#     NLX = NLX[,-c(3,4)]
#     NLX[neg_score] = 1-NLX[pos_score]
#     NLX['Prediction'] = (NLX[pos_score] > 0.5)
#     NLX$Prediction = gsub(TRUE, pos, NLX$Prediction)
#     NLX$Prediction = gsub(FALSE, neg, NLX$Prediction)
#     NLX = NLX[,c(1,4,3,2,5)]
#     write.csv(NLX, paste("~/documents/CPTAC-UCEC/Results/NLX/", ar, f, "/out/Test_slide.csv", sep=''), row.names=FALSE)
#     
#     NLXt = rbind(read.csv(paste("~/documents/CPTAC-UCEC/Results/NL5/", ar, f,"/out/Test_tile.csv", sep='')),
#                  read.csv(paste("~/documents/CPTAC-UCEC/Results/NL8/", ar, f,"/out/Test_tile.csv", sep='')),
#                  read.csv(paste("~/documents/CPTAC-UCEC/Results/NL9/", ar, f,"/out/Test_tile.csv", sep='')))
#     write.csv(NLXt, paste("~/documents/CPTAC-UCEC/Results/NLX/", ar, f, "/out/Test_tile.csv", sep=''), row.names=FALSE)
#     
#   }
# }
# 
# 
# # Multi-level vs. Panoptes tests
# t-test task based sampling
library(ggplot2)
library(ggpubr)
features = c('SL', 'CNVH', 'his', 'MSIst', 'CNVL', 'POLE','FAT1', 'TP53', 'PTEN', 'ZFHX3', 'ARID1A', 'ATM', 'BRCA2', 'CTCF', 'CTNNB1', 'FBXW7', 'JAK1', 'KRAS', 'MTOR', 'PIK3CA', 'PIK3R1', 'PPP2R1A', 'RPL22', 'FGFR2')
arch = c("F1", "F2", "F3", "F4", 'I1', 'I2', 'I3', 'I5', 'I6', 'X1', 'X2', 'X3', 'X4')

all = data.frame(Slide_AUC= numeric(0), Tile_AUC= numeric(0), Architecture= character(0), Feature=character(0), Tile=character(0))
for (a in arch){
  for (nl in c('NL5', 'NL6')){
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
      i = paste(a, f, sep='')
      Test_slide <- read.csv(paste("~/documents/CPTAC-UCEC/Results/", nl, "/", i, "/out/Test_slide.csv", sep=''))
      Test_tile <- read.csv(paste("~/documents/CPTAC-UCEC/Results/", nl, "/", i, "/out/Test_tile.csv", sep=''))

      Sprla = list()
      Sprlb = list()
      for (j in 1:20){
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
      temp_all= data.frame(Patient_AUC=as.numeric(Sprla), Tile_AUC=as.numeric(Sprlb), Architecture=a, Feature=f, Tile=nl)
      all = rbind(all, temp_all)
    }
  }
}

colnames(all) = c("Patient_AUC", "Tile_AUC", "Architecture", "Feature", "Tile")
write.csv(all, file = "~/documents/CPTAC-UCEC/Results/t-test-multi/bootstrap_80%_50.csv", row.names=FALSE)

# 
# # For figure
# library(ggplot2)
# library(ggpubr)
# library(gridExtra)
# all = read.csv("~/documents/CPTAC-UCEC/Results/t-test-multi/bootstrap_80%_50.csv")
# all$Feature <- gsub('his', 'Histology', all$Feature)
# all$Feature <- gsub('SL', 'CNV-H (Endometrioid)', all$Feature)
# all$Feature <- gsub('CNVH', 'CNV-H', all$Feature)
# all$Feature <- gsub('MSIst', 'MSI-high', all$Feature)
# 
# pair = list(c('I6', 'X1'), c('I5', 'X2'))
# 
# for (pwa in pair){
#   wa = pwa[1]
#   wb = pwa[2]
#   all_sub = all[all["Architecture"] == wa | all["Architecture"] == wb, ]
#   all_sub = all_sub[all_sub$Feature %in% c('Histology', 'MSI-high', 'FAT1', 'TP53', 'PTEN', 'ZFHX3', 'CNV-H (Endometrioid)', 'CNV-H'), ]
#   all_sub.a = all_sub[all_sub["Architecture"] == wa & all_sub['Tile'] == "NLX", ]
#   all_sub.b = all_sub[all_sub["Architecture"] == wb & all_sub['Tile'] == "NL5", ]
#   all_sub = rbind(all_sub.a, all_sub.b)
#   
#   pp = ggboxplot(all_sub, x = "Feature", y = "Patient_AUC",
#                  color = "black", fill = "Architecture", palette = "grey")+ 
#     stat_compare_means(method = "t.test", method.args = list(alternative = "greater"), aes(group = Architecture), label = "p.signif", label.y = 1.05)+
#     stat_compare_means(method = "t.test", method.args = list(alternative = "greater"), aes(group = Architecture), label = "p.format", label.y = 1.1)+
#     theme(axis.text.x = element_text(angle = 45, hjust = 1))
#   pl = ggboxplot(all_sub, x = "Feature", y = "Tile_AUC",
#                  color = "black", fill = "Architecture", palette = "grey")+ 
#     stat_compare_means(method = "t.test", method.args = list(alternative = "greater"), aes(group = Architecture), label = "p.signif", label.y = 1.05)+
#     stat_compare_means(method = "t.test", method.args = list(alternative = "greater"), aes(group = Architecture), label = "p.format", label.y = 1.1)+
#     theme(axis.text.x = element_text(angle = 45, hjust = 1))
#   
#   pdf(file=paste("~/documents/CPTAC-UCEC/Results/t-test-multi/", wa, wb, "_lite.pdf", sep=''),
#       width=20,height=5)
#   grid.arrange(pp,pl,nrow=1, ncol=2)
#   dev.off()
# } 
# 
# 
# # For figure barplots
# data_summary <- function(data, varname, groupnames){
#   require(plyr)
#   summary_func <- function(x, col){
#     c(mean = mean(x[[col]], na.rm=TRUE),
#       sd = sd(x[[col]], na.rm=TRUE))
#   }
#   data_sum<-ddply(data, groupnames, .fun=summary_func,
#                   varname)
#   data_sum <- rename(data_sum, c("mean" = varname))
#   return(data_sum)
# }
# 
# library(ggplot2)
# library(ggpubr)
# 
# all = read.csv("~/documents/CPTAC-UCEC/Results/t-test-multi/bootstrap_80%_50.csv")
# all$Feature <- gsub('his', 'Histology', all$Feature)
# all$Feature <- gsub('SL', 'CNV-H (Endometrioid)', all$Feature)
# all$Feature <- gsub('CNVH', 'CNV-H', all$Feature)
# all$Feature <- gsub('MSIst', 'MSI-high', all$Feature)
# all$Architecture = gsub("F1", "Z1", all$Architecture)
# all$Architecture = gsub("F2", "Z2", all$Architecture)
# all$Architecture = gsub("F3", "Z3", all$Architecture)
# all$Architecture = gsub("F4", "Z4", all$Architecture)
# 
# pair = list(c('I6', 'X1'), c('I5', 'X2'))
# 
# for (pwa in pair){
#   wa = pwa[1]
#   wb = pwa[2]
#   all_sub = all[all["Architecture"] == wa | all["Architecture"] == wb, ]
#   all_sub = all_sub[all_sub$Feature %in% c('Histology', 'MSI-high', 'FAT1', 'TP53', 'PTEN', 'ZFHX3', 'CNV-H (Endometrioid)', 'CNV-H'), ]
#   all_sub.a = all_sub[all_sub["Architecture"] == wa & all_sub['Tile'] == "NLX", ]
#   all_sub.b = all_sub[all_sub["Architecture"] == wb & all_sub['Tile'] == "NL5", ]
#   all_sub = rbind(all_sub.a, all_sub.b)
#   
#   all_sub.x = data_summary(all_sub, varname="Patient_AUC", 
#                            groupnames=c("Architecture", "Feature"))
#   all_sub.y = data_summary(all_sub, varname="Tile_AUC", 
#                            groupnames=c("Architecture", "Feature"))
#   
#   pp<- ggplot(all_sub.x, aes(x=Feature, y=Patient_AUC, fill=Architecture)) + 
#     geom_bar(stat="identity", color="black",
#              position=position_dodge()) +
#     scale_fill_manual(values=c("#D3D3D3", "#808080")) +
#     geom_errorbar(aes(ymin=Patient_AUC-sd, ymax=Patient_AUC+sd), width=.2,
#                   position=position_dodge(.9)) +
#     stat_compare_means(data = all_sub, method = "t.test", method.args = list(alternative = "greater"), aes(group = Architecture), label = "p.signif", label.y = 1.05)+
#     stat_compare_means(data = all_sub, method = "t.test", method.args = list(alternative = "greater"), aes(group = Architecture), label = "p.format", label.y = 1.15)+theme_bw()+ 
#     theme(axis.text.x = element_text(angle = 45, hjust = 1), panel.border = element_blank(), panel.grid.major = element_blank(),
#           panel.grid.minor = element_blank(), legend.position = "top")
#   
#   
#   pl<- ggplot(all_sub.y, aes(x=Feature, y=Tile_AUC, fill=Architecture)) + 
#     geom_bar(stat="identity", color="black",
#              position=position_dodge()) +
#     scale_fill_manual(values=c("#D3D3D3", "#808080")) +
#     geom_errorbar(aes(ymin=Tile_AUC-sd, ymax=Tile_AUC+sd), width=.2,
#                   position=position_dodge(.9)) +
#     stat_compare_means(data = all_sub, method = "t.test", method.args = list(alternative = "greater"), aes(group = Architecture), label = "p.signif", label.y = 1.05)+
#     stat_compare_means(data = all_sub, method = "t.test", method.args = list(alternative = "greater"), aes(group = Architecture), label = "p.format", label.y = 1.15)+theme_bw()+ 
#     theme(axis.text.x = element_text(angle = 45, hjust = 1), panel.border = element_blank(), panel.grid.major = element_blank(),
#           panel.grid.minor = element_blank(), legend.position = "top")
#   
#   pdf(file=paste("~/documents/CPTAC-UCEC/Results/t-test-multi/", wa, wb, "_bar_lite.pdf", sep=''),
#       width=20,height=5)
#   grid.arrange(pp,pl,nrow=1, ncol=2)
#   dev.off()
# } 
# 

# 
# #Single 2.5X and 5X with panoptes
# # For figure barplots
# data_summary <- function(data, varname, groupnames){
#   require(plyr)
#   summary_func <- function(x, col){
#     c(mean = mean(x[[col]], na.rm=TRUE),
#       sd = sd(x[[col]], na.rm=TRUE))
#   }
#   data_sum<-ddply(data, groupnames, .fun=summary_func,
#                   varname)
#   data_sum <- rename(data_sum, c("mean" = varname))
#   return(data_sum)
# }
# 
# library(ggplot2)
# library(ggpubr)
# 
# all = read.csv("~/documents/CPTAC-UCEC/Results/t-test-multi/bootstrap_80%_50.csv")
# all$Feature <- gsub('his', 'Histology', all$Feature)
# all$Feature <- gsub('SL', 'CNV-H (Endometrioid)', all$Feature)
# all$Feature <- gsub('CNVH', 'CNV-H', all$Feature)
# all$Feature <- gsub('MSIst', 'MSI-high', all$Feature)
# all$Architecture = gsub("F1", "Z1", all$Architecture)
# all$Architecture = gsub("F2", "Z2", all$Architecture)
# all$Architecture = gsub("F3", "Z3", all$Architecture)
# all$Architecture = gsub("F4", "Z4", all$Architecture)
# 
# pair = list(c('I6', 'X1'), c('I5', 'X2'))
# 
# for (pwa in pair){
#   wa = pwa[1]
#   wb = pwa[2]
#   all_sub = all[all["Architecture"] == wa | all["Architecture"] == wb, ]
#   all_sub = all_sub[all_sub$Feature %in% c('Histology', 'MSI-high', 'FAT1', 'TP53', 'PTEN', 'ZFHX3', 'CNV-H (Endometrioid)', 'CNV-H'), ]
#   
#   all_sub89 = all_sub[all_sub["Tile"] == 'NL9' & all_sub["Architecture"] == wa,]
#   all_sub5 = all_sub[all_sub["Tile"] == 'NL5' & all_sub["Architecture"] == wb,]
#   all_sub = rbind(all_sub89, all_sub5)
#   
#   all_sub.x = data_summary(all_sub, varname="Patient_AUC", 
#                            groupnames=c("Architecture", "Feature"))
#   all_sub.y = data_summary(all_sub, varname="Tile_AUC", 
#                            groupnames=c("Architecture", "Feature"))
#   
#   pp<- ggplot(all_sub.x, aes(x=Feature, y=Patient_AUC, fill=Architecture)) + 
#     geom_bar(stat="identity", color="black",
#              position=position_dodge()) +
#     scale_fill_manual(values=c("#D3D3D3", "#808080")) +
#     geom_errorbar(aes(ymin=Patient_AUC-sd, ymax=Patient_AUC+sd), width=.2,
#                   position=position_dodge(.9)) +
#     stat_compare_means(data = all_sub, method = "t.test", method.args = list(alternative = "greater"), aes(group = Architecture), label = "p.signif", label.y = 1.05)+
#     stat_compare_means(data = all_sub, method = "t.test", method.args = list(alternative = "greater"), aes(group = Architecture), label = "p.format", label.y = 1.15)+theme_bw()+ 
#     theme(axis.text.x = element_text(angle = 45, hjust = 1), panel.border = element_blank(), panel.grid.major = element_blank(),
#           panel.grid.minor = element_blank(), legend.position = "top")
#   
#   
#   pl<- ggplot(all_sub.y, aes(x=Feature, y=Tile_AUC, fill=Architecture)) + 
#     geom_bar(stat="identity", color="black",
#              position=position_dodge()) +
#     scale_fill_manual(values=c("#D3D3D3", "#808080")) +
#     geom_errorbar(aes(ymin=Tile_AUC-sd, ymax=Tile_AUC+sd), width=.2,
#                   position=position_dodge(.9)) +
#     stat_compare_means(data = all_sub, method = "t.test", method.args = list(alternative = "greater"), aes(group = Architecture), label = "p.signif", label.y = 1.05)+
#     stat_compare_means(data = all_sub, method = "t.test", method.args = list(alternative = "greater"), aes(group = Architecture), label = "p.format", label.y = 1.15)+theme_bw()+ 
#     theme(axis.text.x = element_text(angle = 45, hjust = 1), panel.border = element_blank(), panel.grid.major = element_blank(),
#           panel.grid.minor = element_blank(), legend.position = "top")
#   
#   pdf(file=paste("~/documents/CPTAC-UCEC/Results/t-test-multi/NL9/", wa, wb, "_bar_lite.pdf", sep=''),
#       width=20,height=5)
#   grid.arrange(pp,pl,nrow=1, ncol=2)
#   dev.off()
# } 
# 
# 
# # Compare different resolution
# data_summary <- function(data, varname, groupnames){
#   require(plyr)
#   summary_func <- function(x, col){
#     c(mean = mean(x[[col]], na.rm=TRUE),
#       sd = sd(x[[col]], na.rm=TRUE))
#   }
#   data_sum<-ddply(data, groupnames, .fun=summary_func,
#                   varname)
#   return(data_sum)
# }
# 
# 
# library(ggplot2)
# library(ggpubr)
# library(gridExtra)
# 
# for (ar in c("I6", 'I5')){
#   all = read.csv("~/documents/CPTAC-UCEC/Results/t-test-multi/bootstrap_80%_50.csv")
#   all$Feature <- gsub('his', 'Histology', all$Feature)
#   all$Feature <- gsub('SL', 'CNV-H (Endometrioid)', all$Feature)
#   all$Feature <- gsub('CNVH', 'CNV-H', all$Feature)
#   all$Feature <- gsub('MSIst', 'MSI-high', all$Feature)
#   all$Architecture = gsub("F1", "Z1", all$Architecture)
#   all$Architecture = gsub("F2", "Z2", all$Architecture)
#   all$Architecture = gsub("F3", "Z3", all$Architecture)
#   all$Architecture = gsub("F4", "Z4", all$Architecture)
#   all$Tile <- gsub('NL5', '10X', all$Tile)
#   all$Tile <- gsub('NL8', '5X', all$Tile)
#   all$Tile <- gsub('NL9', '2.5X', all$Tile)
#   all$Tile <- gsub('NLX', 'Joint', all$Tile)
#   all.1 = all[all['Tile'] == '2.5X', ]
#   all.2 = all[all['Tile'] == '5X', ]
#   all.3 = all[all['Tile'] == '10X', ]
#   all.4 = all[all['Tile'] == 'Joint', ]
#   all = rbind(all.1, all.2, all.3, all.4)
#   rownames(all) <- NULL
#   all_sub = all[all['Architecture'] == ar, ]
#   all_sub = all_sub[all_sub$Feature %in% c('Histology', 'MSI-high', 'FAT1', 'TP53', 'PTEN', 'ZFHX3', 'CNV-H (Endometrioid)', 'CNV-H'), ]
# 
#   all_sub.x = data_summary(all_sub, varname="Patient_AUC", 
#                            groupnames=c("Tile", "Feature"))
#   all_sub.x  <- rename(all_sub.x , replace = c("mean"="Patient_AUC"))
#   all_sub.y = data_summary(all_sub, varname="Tile_AUC", 
#                            groupnames=c("Tile", "Feature"))
#   all_sub.y  <- rename(all_sub.y , replace = c("mean" = "Tile_AUC"))
#   
#   pp<- ggplot(all_sub.x, aes(x=Feature, y=Patient_AUC, fill=Tile, group=Tile)) + 
#     geom_bar(stat="identity", color="black",
#              position=position_dodge()) +
#     scale_fill_manual(values=c("#D3D3D3", "#939393", "#696969", "#2A2A2A")) +
#     geom_errorbar(aes(ymin=Patient_AUC-sd, ymax=Patient_AUC+sd), width=.2,
#                   position=position_dodge(.9)) + theme_bw()+ 
#     theme(axis.text.x = element_text(angle = 45, hjust = 1), panel.border = element_blank(), panel.grid.major = element_blank(),
#           panel.grid.minor = element_blank(), legend.position = "top")
#   
#   
#   pl<- ggplot(all_sub.y, aes(x=Feature, y=Tile_AUC, fill=Tile, group=Tile)) + 
#     geom_bar(stat="identity", color="black",
#              position=position_dodge()) +
#     scale_fill_manual(values=c("#D3D3D3", "#939393", "#696969", "#2A2A2A")) +
#     geom_errorbar(aes(ymin=Tile_AUC-sd, ymax=Tile_AUC+sd), width=.2,
#                   position=position_dodge(.9)) + theme_bw()+ 
#     theme(axis.text.x = element_text(angle = 45, hjust = 1), panel.border = element_blank(), panel.grid.major = element_blank(),
#           panel.grid.minor = element_blank(), legend.position = "top")
#   
#   pdf(file=paste("~/documents/CPTAC-UCEC/Results/t-test-multi/NLcompare/", ar, "_bar_lite.pdf", sep=''),
#       width=20,height=5)
#   grid.arrange(pp,pl,nrow=1, ncol=2)
#   dev.off()
#  
# }

# mixed vs. independent
## For figure
all = read.csv("~/documents/CPTAC-UCEC/Results/t-test-multi/bootstrap_80%_50.csv")
mixed = list(c('X1', 'Histology'), c('X2', 'CNV-H (Endometrioid)'), c('X3', 'CNV-H'), c('X2', "CNV-L"), c('X1', 'TP53'), c('F1', 'FAT1'), c('I5', 'MSI-high'), c('I5', 'ZFHX3'), c('I2', 'PTEN'), c('I5', 'POLE'), c('F3', 'FGFR2'),
                c('X2', 'MTOR'), c('X3', 'CTCF'), c('I5', 'PIK3R1'), c('X3', 'PIK3CA'), c('I6', 'ARID1A'), c('F1', 'JAK1'), c('I6', 'CTNNB1'), c('F1', 'KRAS'), 
                c('I3', 'FBXW7'), c('I3', 'RPL22'), c('I5', 'BRCA2'), c('I2', 'ATM'), c('X2', 'PPP2R1A'))
ind = list(c('F3', 'Histology'), c('F4', 'CNV-H (Endometrioid)'), c('X4', 'CNV-H'), c('I5', "CNV-L"), c('I1', 'TP53'), c('I5', 'FAT1'), c('X4', 'MSI-high'), c('I3', 'ZFHX3'), c('X3', 'PTEN'), c('F1', 'POLE'), c('F4', 'FGFR2'),
           c('X1', 'MTOR'), c('F2', 'CTCF'), c('X4', 'PIK3R1'), c('I2', 'PIK3CA'), c('X2', 'ARID1A'), c('X3', 'JAK1'), c('I2', 'CTNNB1'), c('I1', 'KRAS'), 
           c('X3', 'FBXW7'), c('X2', 'RPL22'), c('X2', 'BRCA2'), c('X3', 'ATM'), c('I2', 'PPP2R1A'))

all$Feature <- gsub('his', 'Histology', all$Feature)
all$Feature <- gsub('SL', 'CNV-H (Endometrioid)', all$Feature)
all$Feature <- gsub('CNVH', 'CNV-H', all$Feature)
all$Feature <- gsub('MSIst', 'MSI-high', all$Feature)
all$Feature <- gsub('CNVL', 'CNV-L', all$Feature)

colnames(all)[5] <- gsub('Tile', 'Split', colnames(all)[5])
all = all[all$Split == 'NL5' | all$Split == 'NL6',]
all$Split <- gsub('NL5', 'Mixed', all$Split)
all$Split <- gsub('NL6', 'Independent', all$Split)
forfig = data.frame(Patient_AUC= numeric(0), Tile_AUC= numeric(0), Architecture= character(0), Feature=character(0), Split=character(0))
for (mxx in mixed){
  ma = mxx[1]
  mb = mxx[2]
  slt = all[all$Architecture == ma & all$Feature == mb & all$Split == 'Mixed', ]
  forfig = rbind(forfig, slt)
}

for (idd in ind){
  ia = idd[1]
  ib = idd[2]
  slt = all[all$Architecture == ia & all$Feature == ib & all$Split == 'Independent', ]
  forfig = rbind(forfig, slt)
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

pair = c('Mixed', 'Independent')

wa = pair[1]
wb = pair[2]
all_sub = forfig[forfig["Split"] == wa | forfig["Split"] == wb, ]
# all_sub = all_sub[all_sub$Feature %in% c('Histology', 'MSI-high', 'FAT1', 'TP53', 'PTEN', 'ZFHX3', 'CNV-H (Endometrioid)', 'CNV-H'), ]

all_sub.x = data_summary(all_sub, varname="Patient_AUC", 
                         groupnames=c("Split", "Feature"))
all_sub.y = data_summary(all_sub, varname="Tile_AUC", 
                         groupnames=c("Split", "Feature"))

pp<- ggplot(all_sub.x, aes(x=reorder(Feature, -Patient_AUC), y=Patient_AUC, fill=Split)) +
  xlab("") + 
  geom_bar(stat="identity", color="black",
           position=position_dodge())+ coord_cartesian(ylim = c(0,1)) + 
  scale_fill_manual(values=c("#D3D3D3", "#808080")) +
  geom_errorbar(aes(ymin=Patient_AUC-sd, ymax=Patient_AUC+sd), width=.2,
                position=position_dodge(.9)) +
  geom_text(aes(label=round(Patient_AUC, 2)), position=position_dodge(width=0.9), vjust=-2, fontface="bold") + 
  theme_bw()+ 
  theme(axis.text.x = element_text(size = 15, angle = 45, hjust = 1), panel.border = element_blank(), panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), legend.position = "top")+  
  font("xlab", size = 0, color = "black")+
  font("ylab", size = 15, color = "black", face = "bold")+
  font("x.text", size = 15, color = "black", face = "bold")


pl<- ggplot(all_sub.y, aes(x=reorder(Feature, -Tile_AUC), y=Tile_AUC, fill=Split)) + 
  xlab("")+ 
  geom_bar(stat="identity", color="black",
           position=position_dodge()) + coord_cartesian(ylim = c(0,1)) + 
  scale_fill_manual(values=c("#D3D3D3", "#808080")) +
  geom_errorbar(aes(ymin=Tile_AUC-sd, ymax=Tile_AUC+sd), width=.2,
                position=position_dodge(.9)) + 
  geom_text(aes(label=round(Tile_AUC, 2)), position=position_dodge(width=0.9), vjust=-2, fontface="bold") + 
  theme_bw()+ 
  theme(axis.text.x = element_text(size = 15, angle = 45, hjust = 1), panel.border = element_blank(), panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), legend.position = "top")+  
  font("xlab", size = 0, color = "black")+
  font("ylab", size = 15, color = "black", face = "bold")+
  font("x.text", size = 15, color = "black", face = "bold")

pdf(file=paste("~/documents/CPTAC-UCEC/Results/t-test-multi/", wa, wb, "_best_bar_lite.pdf", sep=''),
    width=32,height=7.5)
grid.arrange(pp,pl,nrow=1, ncol=2)
dev.off()


# For figure barplots
data_summary <- function(data, varname, groupnames){
  require(plyr)
  summary_func <- function(x, col){
    c(mean = mean(x[[col]], na.rm=TRUE),
      sd = sd(x[[col]], na.rm=TRUE))
  }
  data_sum<-ddply(data, groupnames, .fun=summary_func,
                  varname)
  data_sum <- plyr::rename(data_sum, c("mean" = varname))
  return(data_sum)
}
arch = list(c('X1', 'Panoptes2'), c('X2', 'Panoptes1'), c('X3', 'Panoptes4'), c('X4', 'Panoptes3'), c('F1', 'Panoptes2_clinical'), c('F2', 'Panoptes1_clinical'), c('F3', 'Panoptes4_clinical'), c('F4', 'Panoptes3_clinical'))
# features = c('CNV-H (Endometrioid)', 'CNV-H', 'Histology', 'MSI-high', 'FAT1', 'TP53', 'PTEN', 'ZFHX3', 'ARID1A', 'ATM', 'BRCA2', 'CTCF', 'CTNNB1', 'FBXW7', 'JAK1', 'KRAS', 'MTOR', 'PIK3CA', 'PIK3R1', 'PPP2R1A', 'RPL22', 'FGFR2')
all = read.csv("~/documents/CPTAC-UCEC/Results/t-test-multi/bootstrap_80%_50.csv")
colnames(all)[5] <- gsub('Tile', 'Split', colnames(all)[5])
all = all[all$Split == 'NL5' | all$Split == 'NL6',]
all$Split <- gsub('NL5', 'Mixed', all$Split)
all$Split <- gsub('NL6', 'Independent', all$Split)

for (arc in arch){
  forfig = data.frame(Patient_AUC= numeric(0), Tile_AUC= numeric(0), Architecture= character(0), Feature=character(0), Split=character(0))
  slt = all[all$Architecture == arc[1], ]
  forfig = rbind(forfig, slt)
  
  all_sub = forfig[forfig["Split"] == 'Mixed' | forfig["Split"] == 'Independent', ]
  # all_sub = all_sub[all_sub$Feature %in% c('Histology', 'MSI-high', 'FAT1', 'TP53', 'PTEN', 'ZFHX3', 'CNV-H (Endometrioid)', 'CNV-H'), ]
  
  all_sub.x = data_summary(all_sub, varname="Patient_AUC", 
                           groupnames=c("Split", "Feature"))
  all_sub.y = data_summary(all_sub, varname="Tile_AUC", 
                           groupnames=c("Split", "Feature"))
  
  pp<- ggplot(all_sub.x, aes(x=reorder(Feature, -Patient_AUC), y=Patient_AUC, fill=Split)) +
    xlab("") + 
    geom_bar(stat="identity", color="black",
             position=position_dodge())+ coord_cartesian(ylim = c(0,1)) + 
    scale_fill_manual(values=c("#D3D3D3", "#808080")) +
    geom_errorbar(aes(ymin=Patient_AUC-sd, ymax=Patient_AUC+sd), width=.2,
                  position=position_dodge(.9)) +
    # geom_text(aes(label=round(Patient_AUC, 2)), position=position_dodge(width=0.9), vjust=-2, size=8) + 
    theme_bw()+ 
    theme(axis.text.x = element_text(size = 22, angle = 45, hjust = 1), panel.border = element_blank(), panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(), legend.position = 'none')+  
    font("xlab", size = 0, color = "black")+
    font("ylab", size = 22, color = "black", face = "bold")+
    font("xy.text", size = 22, color = "black", face = "bold")
  
  pl<- ggplot(all_sub.y, aes(x=reorder(Feature, -Tile_AUC), y=Tile_AUC, fill=Split)) + 
    xlab("")+ 
    geom_bar(stat="identity", color="black",
             position=position_dodge()) + coord_cartesian(ylim = c(0,1)) + 
    scale_fill_manual(values=c("#D3D3D3", "#808080")) +
    geom_errorbar(aes(ymin=Tile_AUC-sd, ymax=Tile_AUC+sd), width=.2,
                  position=position_dodge(.9)) + 
    # geom_text(aes(label=round(Tile_AUC, 2)), position=position_dodge(width=0.9), vjust=-2, size=8) + 
    theme_bw()+ 
    theme(axis.text.x = element_text(size = 22, angle = 45, hjust = 1), panel.border = element_blank(), panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(), legend.position = 'none')+  
    font("xlab", size = 0, color = "black")+
    font("ylab", size = 22, color = "black", face = "bold")+
    font("xy.text", size = 22, color = "black", face = "bold")
  
  pdf(file=paste("~/documents/CPTAC-UCEC/Results/t-test-multi/", wa, wb, "_", arc[2], "_bar_lite.pdf", sep=''),
      width=30,height=7.5)
  grid.arrange(pp,pl,nrow=1, ncol=2)
  dev.off()
}

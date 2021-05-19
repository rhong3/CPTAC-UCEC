# Special
# Calculate bootstraped CI of accuracy, AUROC, AUPRC and other statistical metrics. Gather them into a single file.
library(readxl)
library(caret)
library(pROC)
library(dplyr)
library(MLmetrics)
library(boot)
library(gmodels)
library(stringr)

inlist = c('I1MSI_PMS2_NYU_NL5', 'I1MSI_MSH6_NYU_NL5', 'I1MSI_MSH2_NYU_NL5', 'I1MSI_MLH1_NYU_NL5', 'I1CNVH_P53_NYU_NL5', 'I1TP53_P53_NYU_NL5', 'I1MSI_PMS2-MLH1_NYU_NL5', 'I1MSI_MSH2-MSH6_NYU_NL5', 'I1MSI_PMS2-MLH1-MSH2-MSH6_NYU_NL5', 
           'I1MSI_PMS2_NYU_NL6', 'I1MSI_MSH6_NYU_NL6', 'I1MSI_MSH2_NYU_NL6', 'I1MSI_MLH1_NYU_NL6', 'I1CNVH_P53_NYU_NL6', 'I1TP53_P53_NYU_NL6', 'I1MSI_PMS2-MLH1_NYU_NL6', 'I1MSI_MSH2-MSH6_NYU_NL6', 'I1MSI_PMS2-MLH1-MSH2-MSH6_NYU_NL6',
           'I2MSI_PMS2_NYU_NL5', 'I2MSI_MSH6_NYU_NL5', 'I2MSI_MSH2_NYU_NL5', 'I2MSI_MLH1_NYU_NL5', 'I2CNVH_P53_NYU_NL5', 'I2TP53_P53_NYU_NL5', 'I2MSI_PMS2-MLH1_NYU_NL5', 'I2MSI_MSH2-MSH6_NYU_NL5', 'I2MSI_PMS2-MLH1-MSH2-MSH6_NYU_NL5', 
           'I2MSI_PMS2_NYU_NL6', 'I2MSI_MSH6_NYU_NL6', 'I2MSI_MSH2_NYU_NL6', 'I2MSI_MLH1_NYU_NL6', 'I2CNVH_P53_NYU_NL6', 'I2TP53_P53_NYU_NL6', 'I2MSI_PMS2-MLH1_NYU_NL6', 'I2MSI_MSH2-MSH6_NYU_NL6', 'I2MSI_PMS2-MLH1-MSH2-MSH6_NYU_NL6',
           'I3MSI_PMS2_NYU_NL5', 'I3MSI_MSH6_NYU_NL5', 'I3MSI_MSH2_NYU_NL5', 'I3MSI_MLH1_NYU_NL5', 'I3CNVH_P53_NYU_NL5', 'I3TP53_P53_NYU_NL5', 'I3MSI_PMS2-MLH1_NYU_NL5', 'I3MSI_MSH2-MSH6_NYU_NL5', 'I3MSI_PMS2-MLH1-MSH2-MSH6_NYU_NL5', 
           'I3MSI_PMS2_NYU_NL6', 'I3MSI_MSH6_NYU_NL6', 'I3MSI_MSH2_NYU_NL6', 'I3MSI_MLH1_NYU_NL6', 'I3CNVH_P53_NYU_NL6', 'I3TP53_P53_NYU_NL6', 'I3MSI_PMS2-MLH1_NYU_NL6', 'I3MSI_MSH2-MSH6_NYU_NL6', 'I3MSI_PMS2-MLH1-MSH2-MSH6_NYU_NL6',
           'I5MSI_PMS2_NYU_NL5', 'I5MSI_MSH6_NYU_NL5', 'I5MSI_MSH2_NYU_NL5', 'I5MSI_MLH1_NYU_NL5', 'I5CNVH_P53_NYU_NL5', 'I5TP53_P53_NYU_NL5', 'I5MSI_PMS2-MLH1_NYU_NL5', 'I5MSI_MSH2-MSH6_NYU_NL5', 'I5MSI_PMS2-MLH1-MSH2-MSH6_NYU_NL5', 
           'I5MSI_PMS2_NYU_NL6', 'I5MSI_MSH6_NYU_NL6', 'I5MSI_MSH2_NYU_NL6', 'I5MSI_MLH1_NYU_NL6', 'I5CNVH_P53_NYU_NL6', 'I5TP53_P53_NYU_NL6', 'I5MSI_PMS2-MLH1_NYU_NL6', 'I5MSI_MSH2-MSH6_NYU_NL6', 'I5MSI_PMS2-MLH1-MSH2-MSH6_NYU_NL6',
           'I6MSI_PMS2_NYU_NL5', 'I6MSI_MSH6_NYU_NL5', 'I6MSI_MSH2_NYU_NL5', 'I6MSI_MLH1_NYU_NL5', 'I6CNVH_P53_NYU_NL5', 'I6TP53_P53_NYU_NL5', 'I6MSI_PMS2-MLH1_NYU_NL5', 'I6MSI_MSH2-MSH6_NYU_NL5', 'I6MSI_PMS2-MLH1-MSH2-MSH6_NYU_NL5', 
           'I6MSI_PMS2_NYU_NL6', 'I6MSI_MSH6_NYU_NL6', 'I6MSI_MSH2_NYU_NL6', 'I6MSI_MLH1_NYU_NL6', 'I6CNVH_P53_NYU_NL6', 'I6TP53_P53_NYU_NL6', 'I6MSI_PMS2-MLH1_NYU_NL6', 'I6MSI_MSH2-MSH6_NYU_NL6', 'I6MSI_PMS2-MLH1-MSH2-MSH6_NYU_NL6',
           'X1MSI_PMS2_NYU_NL5', 'X1MSI_MSH6_NYU_NL5', 'X1MSI_MSH2_NYU_NL5', 'X1MSI_MLH1_NYU_NL5', 'X1CNVH_P53_NYU_NL5', 'X1TP53_P53_NYU_NL5', 'X1MSI_PMS2-MLH1_NYU_NL5', 'X1MSI_MSH2-MSH6_NYU_NL5', 'X1MSI_PMS2-MLH1-MSH2-MSH6_NYU_NL5', 
           'X1MSI_PMS2_NYU_NL6', 'X1MSI_MSH6_NYU_NL6', 'X1MSI_MSH2_NYU_NL6', 'X1MSI_MLH1_NYU_NL6', 'X1CNVH_P53_NYU_NL6', 'X1TP53_P53_NYU_NL6', 'X1MSI_PMS2-MLH1_NYU_NL6', 'X1MSI_MSH2-MSH6_NYU_NL6', 'X1MSI_PMS2-MLH1-MSH2-MSH6_NYU_NL6',
           'X2MSI_PMS2_NYU_NL5', 'X2MSI_MSH6_NYU_NL5', 'X2MSI_MSH2_NYU_NL5', 'X2MSI_MLH1_NYU_NL5', 'X2CNVH_P53_NYU_NL5', 'X2TP53_P53_NYU_NL5', 'X2MSI_PMS2-MLH1_NYU_NL5', 'X2MSI_MSH2-MSH6_NYU_NL5', 'X2MSI_PMS2-MLH1-MSH2-MSH6_NYU_NL5', 
           'X2MSI_PMS2_NYU_NL6', 'X2MSI_MSH6_NYU_NL6', 'X2MSI_MSH2_NYU_NL6', 'X2MSI_MLH1_NYU_NL6', 'X2CNVH_P53_NYU_NL6', 'X2TP53_P53_NYU_NL6', 'X2MSI_PMS2-MLH1_NYU_NL6', 'X2MSI_MSH2-MSH6_NYU_NL6', 'X2MSI_PMS2-MLH1-MSH2-MSH6_NYU_NL6',
           'X3MSI_PMS2_NYU_NL5', 'X3MSI_MSH6_NYU_NL5', 'X3MSI_MSH2_NYU_NL5', 'X3MSI_MLH1_NYU_NL5', 'X3CNVH_P53_NYU_NL5', 'X3TP53_P53_NYU_NL5', 'X3MSI_PMS2-MLH1_NYU_NL5', 'X3MSI_MSH2-MSH6_NYU_NL5', 'X3MSI_PMS2-MLH1-MSH2-MSH6_NYU_NL5', 
           'X3MSI_PMS2_NYU_NL6', 'X3MSI_MSH6_NYU_NL6', 'X3MSI_MSH2_NYU_NL6', 'X3MSI_MLH1_NYU_NL6', 'X3CNVH_P53_NYU_NL6', 'X3TP53_P53_NYU_NL6', 'X3MSI_PMS2-MLH1_NYU_NL6', 'X3MSI_MSH2-MSH6_NYU_NL6', 'X3MSI_PMS2-MLH1-MSH2-MSH6_NYU_NL6',
           'X4MSI_PMS2_NYU_NL5', 'X4MSI_MSH6_NYU_NL5', 'X4MSI_MSH2_NYU_NL5', 'X4MSI_MLH1_NYU_NL5', 'X4CNVH_P53_NYU_NL5', 'X4TP53_P53_NYU_NL5', 'X4MSI_PMS2-MLH1_NYU_NL5', 'X4MSI_MSH2-MSH6_NYU_NL5', 'X4MSI_PMS2-MLH1-MSH2-MSH6_NYU_NL5',
           'X4MSI_PMS2_NYU_NL6', 'X4MSI_MSH6_NYU_NL6', 'X4MSI_MSH2_NYU_NL6', 'X4MSI_MLH1_NYU_NL6', 'X4CNVH_P53_NYU_NL6', 'X4TP53_P53_NYU_NL6', 'X4MSI_PMS2-MLH1_NYU_NL6', 'X4MSI_MSH2-MSH6_NYU_NL6', 'X4MSI_PMS2-MLH1-MSH2-MSH6_NYU_NL6',
           'F1MSI_PMS2_NYU_NL5', 'F1MSI_MSH6_NYU_NL5', 'F1MSI_MSH2_NYU_NL5', 'F1MSI_MLH1_NYU_NL5', 'F1CNVH_P53_NYU_NL5', 'F1TP53_P53_NYU_NL5', 'F1MSI_PMS2-MLH1_NYU_NL5', 'F1MSI_MSH2-MSH6_NYU_NL5', 'F1MSI_PMS2-MLH1-MSH2-MSH6_NYU_NL5', 
           'F1MSI_PMS2_NYU_NL6', 'F1MSI_MSH6_NYU_NL6', 'F1MSI_MSH2_NYU_NL6', 'F1MSI_MLH1_NYU_NL6', 'F1CNVH_P53_NYU_NL6', 'F1TP53_P53_NYU_NL6', 'F1MSI_PMS2-MLH1_NYU_NL6', 'F1MSI_MSH2-MSH6_NYU_NL6', 'F1MSI_PMS2-MLH1-MSH2-MSH6_NYU_NL6',
           'F2MSI_PMS2_NYU_NL5', 'F2MSI_MSH6_NYU_NL5', 'F2MSI_MSH2_NYU_NL5', 'F2MSI_MLH1_NYU_NL5', 'F2CNVH_P53_NYU_NL5', 'F2TP53_P53_NYU_NL5', 'F2MSI_PMS2-MLH1_NYU_NL5', 'F2MSI_MSH2-MSH6_NYU_NL5', 'F2MSI_PMS2-MLH1-MSH2-MSH6_NYU_NL5', 
           'F2MSI_PMS2_NYU_NL6', 'F2MSI_MSH6_NYU_NL6', 'F2MSI_MSH2_NYU_NL6', 'F2MSI_MLH1_NYU_NL6', 'F2CNVH_P53_NYU_NL6', 'F2TP53_P53_NYU_NL6', 'F2MSI_PMS2-MLH1_NYU_NL6', 'F2MSI_MSH2-MSH6_NYU_NL6', 'F2MSI_PMS2-MLH1-MSH2-MSH6_NYU_NL6',
           'F3MSI_PMS2_NYU_NL5', 'F3MSI_MSH6_NYU_NL5', 'F3MSI_MSH2_NYU_NL5', 'F3MSI_MLH1_NYU_NL5', 'F3CNVH_P53_NYU_NL5', 'F3TP53_P53_NYU_NL5', 'F3MSI_PMS2-MLH1_NYU_NL5', 'F3MSI_MSH2-MSH6_NYU_NL5', 'F3MSI_PMS2-MLH1-MSH2-MSH6_NYU_NL5', 
           'F3MSI_PMS2_NYU_NL6', 'F3MSI_MSH6_NYU_NL6', 'F3MSI_MSH2_NYU_NL6', 'F3MSI_MLH1_NYU_NL6', 'F3CNVH_P53_NYU_NL6', 'F3TP53_P53_NYU_NL6', 'F3MSI_PMS2-MLH1_NYU_NL6', 'F3MSI_MSH2-MSH6_NYU_NL6', 'F3MSI_PMS2-MLH1-MSH2-MSH6_NYU_NL6',
           'F4MSI_PMS2_NYU_NL5', 'F4MSI_MSH6_NYU_NL5', 'F4MSI_MSH2_NYU_NL5', 'F4MSI_MLH1_NYU_NL5', 'F4CNVH_P53_NYU_NL5', 'F4TP53_P53_NYU_NL5', 'F4MSI_PMS2-MLH1_NYU_NL5', 'F4MSI_MSH2-MSH6_NYU_NL5', 'F4MSI_PMS2-MLH1-MSH2-MSH6_NYU_NL5', 
           'F4MSI_PMS2_NYU_NL6', 'F4MSI_MSH6_NYU_NL6', 'F4MSI_MSH2_NYU_NL6', 'F4MSI_MLH1_NYU_NL6', 'F4CNVH_P53_NYU_NL6', 'F4TP53_P53_NYU_NL6', 'F4MSI_PMS2-MLH1_NYU_NL6', 'F4MSI_MSH2-MSH6_NYU_NL6', 'F4MSI_PMS2-MLH1-MSH2-MSH6_NYU_NL6')

# Check previously calculated trials
previous=read.csv("~/Documents/CPTAC-UCEC/Results/Statistics_NYU_IHC_filter.csv")
existed=paste(paste(previous$Architecture, previous$Feature, sep=''), 'NYU', previous$Tiles, sep='_')
# Find the new trials to be calculated
targets = inlist[which(!inlist %in% existed)]
OUTPUT = setNames(data.frame(matrix(ncol = 51, nrow = 0)), c("Feature", "Architecture", "Tiles", "Patient_ROC.95.CI_lower", "Patient_ROC",                 
                                                             "Patient_ROC.95.CI_upper",      "Patient_PRC.95.CI_lower",      "Patient_PRC",                  "Patient_PRC.95%CI_upper",      "Patient_Accuracy",            
                                                             "Patient_Kappa",                "Patient_AccuracyLower",        "Patient_AccuracyUpper",        "Patient_AccuracyNull",         "Patient_AccuracyPValue",      
                                                             "Patient_McnemarPValue",        "Patient_Sensitivity",          "Patient_Specificity",          "Patient_Pos.Pred.Value",       "Patient_Neg.Pred.Value",      
                                                             "Patient_Precision",            "Patient_Recall",               "Patient_F1",                   "Patient_Prevalence",           "Patient_Detection.Rate",      
                                                             "Patient_Detection.Prevalence", "Patient_Balanced.Accuracy",    "Tile_ROC.95.CI_lower",         "Tile_ROC",                     "Tile_ROC.95%CI_upper",        
                                                             "Tile_PRC.95.CI_lower",         "Tile_PRC",                     "Tile_PRC.95.CI_upper",         "Tile_Accuracy",                "Tile_Kappa",                  
                                                             "Tile_AccuracyLower",           "Tile_AccuracyUpper",           "Tile_AccuracyNull",            "Tile_AccuracyPValue",          "Tile_McnemarPValue",          
                                                             "Tile_Sensitivity",             "Tile_Specificity",             "Tile_Pos.Pred.Value",          "Tile_Neg.Pred.Value",          "Tile_Precision",              
                                                             "Tile_Recall",                  "Tile_F1",                      "Tile_Prevalence",              "Tile_Detection.Rate",          "Tile_Detection.Prevalence",   
                                                             "Tile_Balanced.Accuracy"))

# # PRC function for bootstrap
# auprc = function(data, indices){
#   sampleddf = data[indices,]
#   prc = PRAUC(sampleddf$POS_score, factor(sampleddf$True_label))
#   return(prc)
# }

# Read reference DF
ref = read.csv('~/documents/GYN-HE-IHC/align/final_summary_full.csv')
ref_split = str_split_fixed(ref$IHC_ID, "-", 2)
ref_split[,2] = gsub("1-", "", ref_split[,2])
ref_split[,2] = gsub("2-", "", ref_split[,2])
ref_split = data.frame(ref_split)
colnames(ref_split) = c('Patient', 'IHC')

for (i in targets){
  tryCatch(
    {
      print(i)
      tiles = strsplit(i, '_')[[1]][4]
      tmm = strsplit(i, '_')[[1]][1]
      ihc = strsplit(i, '_')[[1]][2]
      arch = substr(tmm, 1, 2)
      feature = paste(substr(tmm, 3, nchar(tmm)), ihc, sep="_")
      
      # filter
      reflist = c()
      for (ptt in as.character(unique(ref_split$Patient))){
        sublts = ref_split[ref_split$Patient==ptt, ]
        if (all(strsplit(ihc, '-')[[1]] %in% as.character(sublts$IHC))){
          reflist = append(reflist, ptt)
        }
      }
      # # antifilter
      # reflist = as.character(unique(ref_split$Patient))[!as.character(unique(ref_split$Patient)) %in% reflist]
      
      if (strsplit(feature, '_')[[1]][1] == "MSI"){
        pos = 'MSI'
        neg = 'negative'
        POS_score = 'POS_score'
      } else if (strsplit(feature, '_')[[1]][1] == "CNVH"){
        pos = 'Serous-like'
        neg = 'negative'
        POS_score = 'POS_score'
      } else{
        pos = "TP53"
        neg = 'negative'
        POS_score = 'POS_score'
      }
      
      Test_slide <- read.csv(paste("~/documents/CPTAC-UCEC/Results/NYU_IHC_test/", i, "/out/Test_slide.csv", sep=''))
      Test_slide <- Test_slide[Test_slide$slide %in% reflist,]
      Test_tile <- read.csv(paste("~/documents/CPTAC-UCEC/Results/NYU_IHC_test/", i, "/out/Test_tile.csv", sep=''))
      Test_tile <- Test_tile[Test_tile$slide %in% reflist,]
      
      # per patient level
      answers <- factor(Test_slide$True_label)
      results <- factor(Test_slide$Prediction)
      # statistical metrics
      CMP = confusionMatrix(data=results, reference=answers, positive = pos)
      # ROC
      roc =  roc(answers, Test_slide[, POS_score], levels=c(neg, pos))
      rocdf = t(data.frame(ci.auc(roc)))
      colnames(rocdf) = c('ROC.95.CI_lower', 'ROC', 'ROC.95.CI_upper')
      # PRC
      SprcR = PRAUC(Test_slide[, POS_score], factor(Test_slide$True_label))
      Sprls = list()
      for (j in 1:100){
        sampleddf = Test_slide[sample(nrow(Test_slide), round(nrow(Test_slide)*0.9)),]
        Sprc = PRAUC(sampleddf[, POS_score], factor(sampleddf$True_label))
        Sprls[j] = Sprc
      }
      Sprcci = ci(as.numeric(Sprls))
      Sprcdf = data.frame('PRC.95.CI_lower' = Sprcci[2], 'PRC' = SprcR, 'PRC.95.CI_upper' = Sprcci[3])
      # Combine and add prefix
      soverall = cbind(rocdf, Sprcdf, data.frame(t(CMP$overall)), data.frame(t(CMP$byClass)))
      colnames(soverall) = paste('Patient', colnames(soverall), sep='_')
      
      # per tile level
      Tanswers <- factor(Test_tile$True_label)
      Tresults <- factor(Test_tile$Prediction)
      # statistical metrics
      CMT = confusionMatrix(data=Tresults, reference=Tanswers, positive = pos)
      # ROC
      Troc =  roc(Tanswers, Test_tile[, POS_score], levels=c(neg, pos))
      Trocdf = t(data.frame(ci.auc(Troc)))
      colnames(Trocdf) = c('ROC.95.CI_lower', 'ROC', 'ROC.95.CI_upper')
      # PRC
      prcR = PRAUC(Test_tile[, POS_score], factor(Test_tile$True_label))
      prls = list()
      for (j in 1:10){
        sampleddf = Test_tile[sample(nrow(Test_tile), round(nrow(Test_tile)*0.9)),]
        prc = PRAUC(sampleddf[, POS_score], factor(sampleddf$True_label))
        prls[j] = prc
      }
      Tprcci = ci(as.numeric(prls))
      Tprcdf = data.frame('PRC.95.CI_lower' = Tprcci[2], 'PRC' = prcR, 'PRC.95.CI_upper' = Tprcci[3])
      # Combine and add prefix
      Toverall = cbind(Trocdf, Tprcdf, data.frame(t(CMT$overall)), data.frame(t(CMT$byClass)))
      colnames(Toverall) = paste('Tile', colnames(Toverall), sep='_')
      # Key names
      keydf = data.frame("Feature" = feature, "Architecture" = arch, "Tiles" = tiles)
      # combine all df and reset row name
      tempdf = cbind(keydf, soverall, Toverall)
      rownames(tempdf) <- NULL
      OUTPUT = rbind(OUTPUT, tempdf)
    },
    error = function(error_message){
      message(error_message)
      message(i)
      return(NA)
    }
  )  
}

# Bind old with new; sort; save
New_OUTPUT = rbind(previous, OUTPUT)
New_OUTPUT = New_OUTPUT[order(New_OUTPUT$Feature, New_OUTPUT$Tiles, -New_OUTPUT$Patient_ROC),]
write.csv(New_OUTPUT, file = "~/documents/CPTAC-UCEC/Results/Statistics_NYU_IHC_filter.csv", row.names=FALSE)



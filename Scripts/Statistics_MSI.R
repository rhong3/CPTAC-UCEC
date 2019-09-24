# MSI status
# Calculate bootstraped CI of accuracy, AUROC, AUPRC and other statistical metrics. Gather them into a single file.
library(readxl)
library(caret)
library(pROC)
library(dplyr)
library(MLmetrics)
library(boot)
library(gmodels)


# Check previously calculated trials
previous=read.csv("~/Documents/CPTAC-UCEC/Results/Statistics_MSI.csv")
existed=paste(previous$Tiles, paste(previous$Architecture, 'MSIst', sep=''), sep='/')
sum=read_excel('~/documents/CPTAC-UCEC/Results/Summary.xlsx', sheet = 4)
inlist = paste(sum$Tiles, paste(sum$Architecture, 'MSIst', sep=''), sep='/')
# Find the new trials to be calculated
targets = inlist[which(!inlist %in% existed)]
OUTPUT = setNames(data.frame(matrix(ncol = 50, nrow = 0)), c("Architecture", "Tiles", "Patient_ROC.95.CI_lower", "Patient_ROC",                 
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

for (i in targets){
  tryCatch(
    {
      print(i)
      folder = strsplit(i, '-')[[1]][1]  #split replicated trials
      tiles = strsplit(folder, '/')[[1]][1]  #get NL number
      folder_name = strsplit(folder, '/')[[1]][2]  #get folder name
      arch = substr(folder_name, 1, 2)  #get architecture used
      Test_slide <- read.csv(paste("~/documents/CPTAC-UCEC/Results/", i, "/out/Test_slide.csv", sep=''))
      Test_tile <- read.csv(paste("~/documents/CPTAC-UCEC/Results/", i, "/out/Test_tile.csv", sep=''))
      
      # per patient level
      answers <- factor(Test_slide$True_label)
      results <- factor(Test_slide$Prediction)
      # statistical metrics
      CMP = confusionMatrix(data=results, reference=answers, positive = 'MSI-H')
      # ROC
      roc =  roc(answers, Test_slide$MSI.H_score, levels=c('MSS', 'MSI-H'))
      rocdf = t(data.frame(ci.auc(roc)))
      colnames(rocdf) = c('ROC.95.CI_lower', 'ROC', 'ROC.95.CI_upper')
      # PRC
      SprcR = PRAUC(Test_slide$MSI.H_score, factor(Test_slide$True_label))
      Sprls = list()
      for (i in 1:100){
        sampleddf = Test_slide[sample(nrow(Test_slide), round(nrow(Test_slide)*0.8)),]
        Sprc = PRAUC(sampleddf$MSI.H_score, factor(sampleddf$True_label))
        Sprls[i] = Sprc
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
      CMT = confusionMatrix(data=Tresults, reference=Tanswers)
      # ROC
      Troc =  roc(Tanswers, Test_tile$MSI.H_score, levels=c('MSS', 'MSI-H'))
      Trocdf = t(data.frame(ci.auc(Troc)))
      colnames(Trocdf) = c('ROC.95.CI_lower', 'ROC', 'ROC.95.CI_upper')
      # PRC
      prcR = PRAUC(Test_tile$MSI.H_score, factor(Test_tile$True_label))
      prls = list()
      for (i in 1:10){
        sampleddf = Test_tile[sample(nrow(Test_tile), round(nrow(Test_tile)*0.8)),]
        prc = PRAUC(sampleddf$MSI.H_score, factor(sampleddf$True_label))
        prls[i] = prc
      }
      Tprcci = ci(as.numeric(prls))
      Tprcdf = data.frame('PRC.95.CI_lower' = Tprcci[2], 'PRC' = prcR, 'PRC.95.CI_upper' = Tprcci[3])
      # Combine and add prefix
      Toverall = cbind(Trocdf, Tprcdf, data.frame(t(CMT$overall)), data.frame(t(CMT$byClass)))
      colnames(Toverall) = paste('Tile', colnames(Toverall), sep='_')
      # Key names
      keydf = data.frame("Architecture" = arch, Tiles = tiles)
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
New_OUTPUT = New_OUTPUT[order(New_OUTPUT$Tiles, New_OUTPUT$Architecture),]
write.csv(New_OUTPUT, file = "~/documents/CPTAC-UCEC/Results/Statistics_MSI.csv", row.names=FALSE)



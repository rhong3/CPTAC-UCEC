# Calculate bootstraped CI of accuracy, AUROC, AUPRC and other statistical metrics. Gather them into a single file.
library(readxl)
library(caret)
library(pROC)
library(dplyr)
library(MLmetrics)
library(boot)
Sys.setenv('R_MAX_VSIZE'=10000000000000000000)

# Mutations
sum=read_excel('~/documents/CPTAC-UCEC/Results/Summary.xlsx', sheet = 2)
targets = paste(sum$Tiles, paste(sum$Architecture, sum$Mutation, sep=''), sep='/')
OUTPUT = setNames(data.frame(matrix(ncol = 51, nrow = 0)), c("Mutation", "Architecture", "Tiles", "Patient_ROC.95%CI_lower", "Patient_ROC",                 
                                                             "Patient_ROC.95%CI_upper",      "Patient_PRC.95.CI_lower",      "Patient_PRC",                  "Patient_PRC.95.CI_upper",      "Patient_Accuracy",            
                                                              "Patient_Kappa",                "Patient_AccuracyLower",        "Patient_AccuracyUpper",        "Patient_AccuracyNull",         "Patient_AccuracyPValue",      
                                                              "Patient_McnemarPValue",        "Patient_Sensitivity",          "Patient_Specificity",          "Patient_Pos.Pred.Value",       "Patient_Neg.Pred.Value",      
                                                              "Patient_Precision",            "Patient_Recall",               "Patient_F1",                   "Patient_Prevalence",           "Patient_Detection.Rate",      
                                                              "Patient_Detection.Prevalence", "Patient_Balanced.Accuracy",    "Tile_ROC.95%CI_lower",         "Tile_ROC",                     "Tile_ROC.95%CI_upper",        
                                                              "Tile_PRC.95.CI_lower",         "Tile_PRC",                     "Tile_PRC.95.CI_upper",         "Tile_Accuracy",                "Tile_Kappa",                  
                                                              "Tile_AccuracyLower",           "Tile_AccuracyUpper",           "Tile_AccuracyNull",            "Tile_AccuracyPValue",          "Tile_McnemarPValue",          
                                                              "Tile_Sensitivity",             "Tile_Specificity",             "Tile_Pos.Pred.Value",          "Tile_Neg.Pred.Value",          "Tile_Precision",              
                                                              "Tile_Recall",                  "Tile_F1",                      "Tile_Prevalence",              "Tile_Detection.Rate",          "Tile_Detection.Prevalence",   
                                                              "Tile_Balanced.Accuracy"))

for (i in targets){
  tryCatch(
    {
      folder = strsplit(i, '-')[[1]][1]  #split replicated trials
      tiles = strsplit(folder, '/')[[1]][1]  #get NL number
      folder_name = strsplit(folder, '/')[[1]][2]  #get folder name
      pos = substr(folder_name, 3, nchar(folder_name))  #get mutation
      arch = substr(folder_name, 1, 2)  #get architecture used
      Test_slide <- read.csv(paste("~/documents/CPTAC-UCEC/Results/", i, "/out/Test_slide.csv", sep=''))
      Test_tile <- read.csv(paste("~/documents/CPTAC-UCEC/Results/", i, "/out/Test_tile.csv", sep=''))
      
      # PRC function for bootstrap
      auprc = function(data, indices){
        sampleddf = data[indices,]
        prc = PRAUC(sampleddf$POS_score, factor(sampleddf$True_label))
        return(prc)
      }
      
      # per patient level
      answers <- factor(Test_slide$True_label)
      results <- factor(Test_slide$Prediction)
      # statistical metrics
      CMP = confusionMatrix(data=results, reference=answers, positive = pos)
      # ROC
      roc =  roc(answers, Test_slide$POS_score)
      rocdf = t(data.frame(ci.auc(roc)))
      colnames(rocdf) = c('ROC.95%CI_lower', 'ROC', 'ROC.95%CI_upper')
      # PRC
      prcci=boot.ci(boot(data = Test_slide, statistic=auprc, R=1000), type="bca")
      prcdf = data.frame('PRC.95%CI_lower' = prcci$bca[4], 'PRC' = prcci$t0, 'PRC.95%CI_upper' = prcci$bca[5])
      # Combine and add prefix
      soverall = cbind(rocdf, prcdf, data.frame(t(CMP$overall)), data.frame(t(CMP$byClass)))
      colnames(soverall) = paste('Patient', colnames(soverall), sep='_')
      
      # per tile level
      Tanswers <- factor(Test_tile$True_label)
      Tresults <- factor(Test_tile$Prediction)
      # statistical metrics
      CMT = confusionMatrix(data=Tresults, reference=Tanswers)
      # ROC
      Troc =  roc(Tanswers, Test_tile$POS_score)
      Trocdf = t(data.frame(ci.auc(Troc)))
      colnames(Trocdf) = c('ROC.95%CI_lower', 'ROC', 'ROC.95%CI_upper')
      # PRC
      Tprcci=boot.ci(boot(data = Test_tile, statistic=auprc, R=4), type="bca")
      Tprcdf = data.frame('PRC.95%CI_lower' = Tprcci$bca[4], 'PRC' = Tprcci$t0, 'PRC.95%CI_upper' = Tprcci$bca[5])
      # Combine and add prefix
      Toverall = cbind(Trocdf, Tprcdf, data.frame(t(CMT$overall)), data.frame(t(CMT$byClass)))
      colnames(Toverall) = paste('Tile', colnames(Toverall), sep='_')
      # Key names
      keydf = data.frame("Mutation" = pos, "Architecture" = arch, Tiles = tiles)
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

write.csv(OUTPUT, file = "~/documents/CPTAC-UCEC/Results/Statistics_mutations.csv")

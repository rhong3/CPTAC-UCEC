# Subtypes
# Calculate bootstraped CI of accuracy, AUROC, AUPRC and other statistical metrics. Gather them into a single file.
library(readxl)
library(caret)
library(pROC)
library(dplyr)
library(MLmetrics)
library(boot)
library(gmodels)


previous=read.csv("~/Documents/CPTAC-UCEC/Results/Statistics_subtype.csv")
existed=paste(previous$Tiles, paste(previous$Architecture, "ST", sep=''), sep='/')
sum=read_excel('~/documents/CPTAC-UCEC/Results/Summary.xlsx', sheet = 1)
inlist = paste(sum$Tiles, paste(sum$Architecture, "ST", sep=''), sep='/')
# Find the new trials to be calculated
targets = inlist[which(!inlist %in% existed)]
OUTPUT = setNames(data.frame(matrix(ncol = 158, nrow = 0)), c(  "Architecture"                             , "Tiles"                                    , "Patient_Multiclass_ROC.95.CI_lower"   ,    
                                                                "Patient_Multiclass_ROC"                   , "Patient_Multiclass_ROC.95.CI_upper"       , "Patient_MSI_ROC.95.CI_lower"          ,    
                                                                "Patient_MSI_ROC"                          , "Patient_MSI_ROC.95.CI_upper"              , "Patient_Endometrioid_ROC.95.CI_lower" ,    
                                                                "Patient_Endometrioid_ROC"                 , "Patient_Endometrioid_ROC.95.CI_upper"     , "Patient_Serous.like_ROC.95.CI_lower"  ,    
                                                                "Patient_Serous.like_ROC"                  , "Patient_Serous.like_ROC.95.CI_upper"      , "Patient_POLE_ROC.95.CI_lower"         ,    
                                                                "Patient_POLE_ROC"                         , "Patient_POLE_ROC.95.CI_upper"             , "Patient_MSI_PRC.95.CI_lower"          ,    
                                                                "Patient_MSI_PRC"                          , "Patient_MSI_PRC.95.CI_upper"              , "Patient_Endometrioid_PRC.95.CI_lower" ,    
                                                                "Patient_Endometrioid_PRC"                 , "Patient_Endometrioid_PRC.95.CI_upper"     , "Patient_Serous.like_PRC.95.CI_lower"  ,    
                                                                "Patient_Serous.like_PRC"                  , "Patient_Serous.like_PRC.95.CI_upper"      , "Patient_POLE_PRC.95.CI_lower"         ,    
                                                                "Patient_POLE_PRC"                         , "Patient_POLE_PRC.95.CI_upper"             , "Patient_Accuracy"                     ,    
                                                                "Patient_Kappa"                            , "Patient_AccuracyLower"                    , "Patient_AccuracyUpper"                ,    
                                                                "Patient_AccuracyNull"                     , "Patient_AccuracyPValue"                   , "Patient_McnemarPValue"                ,    
                                                                "Patient_Endometrioid_Sensitivity"         , "Patient_Endometrioid_Specificity"         , "Patient_Endometrioid_Pos.Pred.Value"  ,    
                                                                "Patient_Endometrioid_Neg.Pred.Value"      , "Patient_Endometrioid_Precision"           , "Patient_Endometrioid_Recall"          ,    
                                                                "Patient_Endometrioid_F1"                  , "Patient_Endometrioid_Prevalence"          , "Patient_Endometrioid_Detection.Rate"  ,    
                                                                "Patient_Endometrioid_Detection.Prevalence", "Patient_Endometrioid_Balanced.Accuracy"   , "Patient_MSI_Sensitivity"              ,    
                                                                "Patient_MSI_Specificity"                  , "Patient_MSI_Pos.Pred.Value"               , "Patient_MSI_Neg.Pred.Value"           ,    
                                                                "Patient_MSI_Precision"                    , "Patient_MSI_Recall"                       , "Patient_MSI_F1"                       ,    
                                                                "Patient_MSI_Prevalence"                   , "Patient_MSI_Detection.Rate"               , "Patient_MSI_Detection.Prevalence"     ,    
                                                                "Patient_MSI_Balanced.Accuracy"            , "Patient_POLE_Sensitivity"                 , "Patient_POLE_Specificity"             ,    
                                                                "Patient_POLE_Pos.Pred.Value"              , "Patient_POLE_Neg.Pred.Value"              , "Patient_POLE_Precision"               ,    
                                                                "Patient_POLE_Recall"                      , "Patient_POLE_F1"                          , "Patient_POLE_Prevalence"              ,    
                                                                "Patient_POLE_Detection.Rate"              , "Patient_POLE_Detection.Prevalence"        , "Patient_POLE_Balanced.Accuracy"       ,    
                                                                "Patient_Serous.like_Sensitivity"          , "Patient_Serous.like_Specificity"          , "Patient_Serous.like_Pos.Pred.Value"   ,    
                                                                "Patient_Serous.like_Neg.Pred.Value"       , "Patient_Serous.like_Precision"            , "Patient_Serous.like_Recall"           ,    
                                                                "Patient_Serous.like_F1"                   , "Patient_Serous.like_Prevalence"           , "Patient_Serous.like_Detection.Rate"   ,    
                                                                "Patient_Serous.like_Detection.Prevalence" , "Patient_Serous.like_Balanced.Accuracy"    , "Tile_Multiclass_ROC.95.CI_lower"      ,    
                                                                "Tile_Multiclass_ROC"                      , "Tile_Multiclass_ROC.95.CI_upper"          , "Tile_MSI_ROC.95.CI_lower"             ,    
                                                                "Tile_MSI_ROC"                             , "Tile_MSI_ROC.95.CI_upper"                 , "Tile_Endometrioid_ROC.95.CI_lower"    ,    
                                                                "Tile_Endometrioid_ROC"                    , "Tile_Endometrioid_ROC.95.CI_upper"        , "Tile_Serous.like_ROC.95.CI_lower"     ,    
                                                                "Tile_Serous.like_ROC"                     , "Tile_Serous.like_ROC.95.CI_upper"         , "Tile_POLE_ROC.95.CI_lower"            ,    
                                                                "Tile_POLE_ROC"                            , "Tile_POLE_ROC.95.CI_upper"                , "Tile_MSI_PRC.95.CI_lower"             ,    
                                                                "Tile_MSI_PRC"                             , "Tile_MSI_PRC.95.CI_upper"                 , "Tile_Endometrioid_PRC.95.CI_lower"    ,    
                                                                "Tile_Endometrioid_PRC"                   ,  "Tile_Endometrioid_PRC.95.CI_upper"       ,  "Tile_Serous.like_PRC.95.CI_lower"    ,     
                                                                "Tile_Serous.like_PRC"                    ,  "Tile_Serous.like_PRC.95.CI_upper"        ,  "Tile_POLE_PRC.95.CI_lower"           ,     
                                                                "Tile_POLE_PRC"                           ,  "Tile_POLE_PRC.95.CI_upper"               ,  "Tile_Accuracy"                       ,     
                                                                "Tile_Kappa"                              ,  "Tile_AccuracyLower"                      ,  "Tile_AccuracyUpper"                  ,     
                                                                "Tile_AccuracyNull"                       ,  "Tile_AccuracyPValue"                     ,  "Tile_McnemarPValue"                  ,     
                                                                "Tile_Endometrioid_Sensitivity"           ,  "Tile_Endometrioid_Specificity"           ,  "Tile_Endometrioid_Pos.Pred.Value"    ,     
                                                                "Tile_Endometrioid_Neg.Pred.Value"        ,  "Tile_Endometrioid_Precision"             ,  "Tile_Endometrioid_Recall"            ,     
                                                                "Tile_Endometrioid_F1"                    ,  "Tile_Endometrioid_Prevalence"            ,  "Tile_Endometrioid_Detection.Rate"    ,     
                                                                "Tile_Endometrioid_Detection.Prevalence"  ,  "Tile_Endometrioid_Balanced.Accuracy"     ,  "Tile_MSI_Sensitivity"                ,     
                                                                "Tile_MSI_Specificity"                    ,  "Tile_MSI_Pos.Pred.Value"                 ,  "Tile_MSI_Neg.Pred.Value"             ,     
                                                                "Tile_MSI_Precision"                      ,  "Tile_MSI_Recall"                         ,  "Tile_MSI_F1"                         ,     
                                                                "Tile_MSI_Prevalence"                     ,  "Tile_MSI_Detection.Rate"                 ,  "Tile_MSI_Detection.Prevalence"       ,     
                                                                "Tile_MSI_Balanced.Accuracy"              ,  "Tile_POLE_Sensitivity"                   ,  "Tile_POLE_Specificity"               ,     
                                                                "Tile_POLE_Pos.Pred.Value"                ,  "Tile_POLE_Neg.Pred.Value"                ,  "Tile_POLE_Precision"                 ,     
                                                                "Tile_POLE_Recall"                        ,  "Tile_POLE_F1"                            ,  "Tile_POLE_Prevalence"                ,     
                                                                "Tile_POLE_Detection.Rate"                ,  "Tile_POLE_Detection.Prevalence"          ,  "Tile_POLE_Balanced.Accuracy"         ,     
                                                                "Tile_Serous.like_Sensitivity"            ,  "Tile_Serous.like_Specificity"            ,  "Tile_Serous.like_Pos.Pred.Value"     ,     
                                                                "Tile_Serous.like_Neg.Pred.Value"         ,  "Tile_Serous.like_Precision"              ,  "Tile_Serous.like_Recall"             ,     
                                                                "Tile_Serous.like_F1"                     ,  "Tile_Serous.like_Prevalence"             ,  "Tile_Serous.like_Detection.Rate"     ,     
                                                                "Tile_Serous.like_Detection.Prevalence"   ,  "Tile_Serous.like_Balanced.Accuracy"))



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
      CMP = confusionMatrix(data=results, reference=answers)
      dddf = data.frame(t(CMP$overall))
      for (m in 1:4){
        temmp = data.frame(t(CMP$byClass[m,]))
        colnames(temmp) = paste(gsub('-', '\\.', strsplit(rownames(CMP$byClass)[m], ': ')[[1]][2]), colnames(temmp), sep='_')
        dddf = cbind(dddf, temmp)
      }
      
      # multiclass ROC
      score = select(Test_slide, MSI_score, Endometrioid_score, Serous.like_score, POLE_score)
      colnames(score) = c("MSI", "Endometrioid", "Serous-like", "POLE")
      roc =  multiclass.roc(answers, score)$auc
      rocls=list()
      for (q in 1:100){
        sampleddf = Test_slide[sample(nrow(Test_slide), round(nrow(Test_slide)*0.8)),]
        score = select(sampleddf, MSI_score, Endometrioid_score, Serous.like_score, POLE_score)
        colnames(score) = c("MSI", "Endometrioid", "Serous-like", "POLE") 
        answers <- factor(sampleddf$True_label)
        rocls[q] = multiclass.roc(answers, score)$auc
      }
      rocci = ci(as.numeric(rocls))
      mcroc = data.frame('Multiclass_ROC.95.CI_lower' = rocci[2], 'Multiclass_ROC' = roc, 'Multiclass_ROC.95.CI_upper' = rocci[3])
      
      rocccc = as.data.frame(matrix(0, ncol = 0, nrow = 1))
      prcccc = as.data.frame(matrix(0, ncol = 0, nrow = 1))
      #ROC and PRC
      for (w in (length(colnames(Test_slide))-5):(length(colnames(Test_slide))-2)){
        cpTest_slide = Test_slide
        case=strsplit(colnames(cpTest_slide)[w], '_')[[1]][1]
        case = gsub('\\.', '-', c(case))
        cpTest_slide$True_label = as.character(cpTest_slide$True_label)
        cpTest_slide$True_label[cpTest_slide$True_label != case] = "negative"
        cpTest_slide$True_label = as.factor(cpTest_slide$True_label)
        answers <- factor(cpTest_slide$True_label)
        
        #ROC
        roc =  roc(answers, cpTest_slide[,w], levels = c("negative", case))
        rocdf = t(data.frame(ci.auc(roc)))
        colnames(rocdf) = paste(gsub('-', '\\.', c(case)), c('ROC.95.CI_lower', 'ROC', 'ROC.95.CI_upper'), sep='_')
        rocccc = cbind(rocccc, rocdf)
        
        # PRC
        SprcR = PRAUC(cpTest_slide[,w], answers)
        Sprls = list()
        for (j in 1:100){
          sampleddf = cpTest_slide[sample(nrow(cpTest_slide), round(nrow(cpTest_slide)*0.95)),]
          Sprc = PRAUC(sampleddf[,w], factor(sampleddf$True_label))
          Sprls[j] = Sprc
        }
        Sprcci = ci(as.numeric(Sprls))
        Sprcdf = data.frame('PRC.95.CI_lower' = Sprcci[2], 'PRC' = SprcR, 'PRC.95.CI_upper' = Sprcci[3])
        colnames(Sprcdf) = paste(gsub('-', '\\.', c(case)), colnames(Sprcdf), sep='_')
        prcccc = cbind(prcccc, Sprcdf)
      }
      
      # Combine and add prefix
      soverall = cbind(mcroc, rocccc, prcccc, dddf)
      colnames(soverall) = paste('Patient', colnames(soverall), sep='_')
      
      
      
      # per tile level
      answers <- factor(Test_tile$True_label)
      results <- factor(Test_tile$Prediction)
      # statistical metrics
      CMT = confusionMatrix(data=results, reference=answers)
      Tdddf = data.frame(t(CMT$overall))
      for (m in 1:4){
        Ttemmp = data.frame(t(CMT$byClass[m,]))
        colnames(Ttemmp) = paste(gsub('-', '\\.', strsplit(rownames(CMT$byClass)[m], ': ')[[1]][2]), colnames(Ttemmp), sep='_')
        Tdddf = cbind(Tdddf, Ttemmp)
      }
      
      # multiclass ROC
      Tscore = select(Test_tile, MSI_score, Endometrioid_score, Serous.like_score, POLE_score)
      colnames(Tscore) = c("MSI", "Endometrioid", "Serous-like", "POLE")
      Troc =  multiclass.roc(answers, Tscore)$auc
      Trocls=list()
      for (q in 1:10){
        Tsampleddf = Test_tile[sample(nrow(Test_tile), round(nrow(Test_tile)*0.8)),]
        Tscore = select(Tsampleddf, MSI_score, Endometrioid_score, Serous.like_score, POLE_score)
        colnames(Tscore) = c("MSI", "Endometrioid", "Serous-like", "POLE") 
        Tanswers <- factor(Tsampleddf$True_label)
        Trocls[q] = multiclass.roc(Tanswers, Tscore)$auc
      }
      Trocci = ci(as.numeric(Trocls))
      Tmcroc = data.frame('Multiclass_ROC.95.CI_lower' = Trocci[2], 'Multiclass_ROC' = Troc, 'Multiclass_ROC.95.CI_upper' = Trocci[3])
      
      Trocccc = as.data.frame(matrix(0, ncol = 0, nrow = 1))
      Tprcccc = as.data.frame(matrix(0, ncol = 0, nrow = 1))
      #ROC and PRC
      for (w in (length(colnames(Test_tile))-5):(length(colnames(Test_tile))-2)){
        cpTest_tile = Test_tile
        case=strsplit(colnames(cpTest_tile)[w], '_')[[1]][1]
        case = gsub('\\.', '-', c(case))
        cpTest_tile$True_label = as.character(cpTest_tile$True_label)
        cpTest_tile$True_label[cpTest_tile$True_label != case] = "negative"
        cpTest_tile$True_label = as.factor(cpTest_tile$True_label)
        Tanswers <- factor(cpTest_tile$True_label)
        
        #ROC
        Troc =  roc(Tanswers, cpTest_tile[,w], levels = c("negative", case))
        Trocdf = t(data.frame(ci.auc(Troc)))
        colnames(Trocdf) = paste(gsub('-', '\\.', c(case)), c('ROC.95.CI_lower', 'ROC', 'ROC.95.CI_upper'), sep='_')
        Trocccc = cbind(Trocccc, Trocdf)
        
        # PRC
        TprcR = PRAUC(cpTest_tile[,w], Tanswers)
        Tprls = list()
        for (j in 1:10){
          sampleddf = cpTest_tile[sample(nrow(cpTest_tile), round(nrow(cpTest_tile)*0.95)),]
          Tprc = PRAUC(sampleddf[,w], factor(sampleddf$True_label))
          Tprls[j] = Tprc
        }
        Tprcci = ci(as.numeric(Tprls))
        Tprcdf = data.frame('PRC.95.CI_lower' = Tprcci[2], 'PRC' = TprcR, 'PRC.95.CI_upper' = Tprcci[3])
        colnames(Tprcdf) = paste(gsub('-', '\\.', c(case)), colnames(Tprcdf), sep='_')
        Tprcccc = cbind(Tprcccc, Tprcdf)
      }
      
      # Combine and add prefix
      Toverall = cbind(Tmcroc, Trocccc, Tprcccc, Tdddf)
      colnames(Toverall) = paste('Tile', colnames(Toverall), sep='_')
      
      # Key names
      keydf = data.frame("Architecture" = arch, "Tiles" = tiles)
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
write.csv(New_OUTPUT, file = "~/documents/CPTAC-UCEC/Results/Statistics_subtype.csv", row.names=FALSE)

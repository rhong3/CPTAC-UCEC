# Combined POLE predictor
library(readxl)
library(caret)
library(pROC)
library(dplyr)
library(MLmetrics)
library(boot)
library(gmodels)

generate=function(arch1, arch2, arch3){
  CNVL_slide = read.csv(paste("~/Documents/CPTAC-UCEC/Results/NL6/", arch1, "CNVL/out/Test_slide.csv", sep=""))
  CNVL_slide = CNVL_slide[, c('slide', 'NEG_score', 'True_label')]
  CNVL_slide$True_label = as.numeric(CNVL_slide$True_label != "negative")
  CNVH_slide = read.csv(paste("~/Documents/CPTAC-UCEC/Results/NL6/", arch2, "CNVH/out/Test_slide.csv", sep=""))
  CNVH_slide = CNVH_slide[, c('slide', 'NEG_score', 'True_label')]
  CNVH_slide$True_label = as.numeric(CNVH_slide$True_label != "negative")
  MSI_slide = read.csv(paste("~/Documents/CPTAC-UCEC/Results/NL6/", arch3, "MSIst/out/Test_slide.csv", sep=""))
  MSI_slide = MSI_slide[, c('slide', 'MSS_score', 'True_label')]
  MSI_slide$True_label = as.numeric(MSI_slide$True_label != "MSS")
  
  mg_slide = inner_join(inner_join(CNVH_slide, CNVL_slide, by="slide", suffix=c(".CNVH", ".CNVL")), MSI_slide, by="slide")
  mg_slide$POS_score = (mg_slide$NEG_score.CNVH + mg_slide$NEG_score.CNVL + mg_slide$MSS_score)/3
  mg_slide$True_label = as.numeric((mg_slide$True_label.CNVH + mg_slide$True_label.CNVL + mg_slide$True_label) == 0)
  mg_slide = mg_slide[, c('slide', 'POS_score', 'True_label')]
  mg_slide$NEG_score = 1-mg_slide$POS_score
  mg_slide$Prediction = as.numeric(mg_slide$POS_score>0.5)
  mg_slide$True_label = gsub(1, "POLE", mg_slide$True_label)
  mg_slide$True_label = gsub(0, "negative", mg_slide$True_label)
  mg_slide$Prediction = gsub(1, "POLE", mg_slide$Prediction)
  mg_slide$Prediction = gsub(0, "negative", mg_slide$Prediction)
  
  CNVL_tile = read.csv(paste("~/Documents/CPTAC-UCEC/Results/NL6/", arch1, "CNVL/out/Test_tile.csv", sep=""))
  colnames(CNVL_tile) = gsub("L0path", "path", colnames(CNVL_tile))
  CNVL_tile = CNVL_tile[, c('slide', 'path', 'NEG_score', 'True_label')]
  CNVL_tile$True_label = as.numeric(CNVL_tile$True_label != "negative")
  CNVH_tile = read.csv(paste("~/Documents/CPTAC-UCEC/Results/NL6/", arch2, "CNVH/out/Test_tile.csv", sep=""))
  colnames(CNVH_tile) = gsub("L0path", "path", colnames(CNVH_tile))
  CNVH_tile = CNVH_tile[, c('slide', 'path', 'NEG_score', 'True_label')]
  CNVH_tile$True_label = as.numeric(CNVH_tile$True_label != "negative")
  MSI_tile = read.csv(paste("~/Documents/CPTAC-UCEC/Results/NL6/", arch3, "MSIst/out/Test_tile.csv", sep=""))
  colnames(MSI_tile) = gsub("L0path", "path", colnames(MSI_tile))
  MSI_tile = MSI_tile[, c('slide', 'path', 'MSS_score', 'True_label')]
  MSI_tile$True_label = as.numeric(MSI_tile$True_label != "MSS")
  
  mg_tile = inner_join(inner_join(CNVH_tile, CNVL_tile, by=c("slide", "path"), suffix=c(".CNVH", ".CNVL")), MSI_tile, by=c("slide", "path"))
  mg_tile$POS_score = (mg_tile$NEG_score.CNVH + mg_tile$NEG_score.CNVL + mg_tile$MSS_score)/3
  mg_tile$True_label = as.numeric((mg_tile$True_label.CNVH + mg_tile$True_label.CNVL + mg_tile$True_label) == 0)
  mg_tile = mg_tile[, c('slide', 'path', 'POS_score', 'True_label')]
  mg_tile$NEG_score = 1-mg_tile$POS_score
  mg_tile$Prediction = as.numeric(mg_tile$POS_score>0.5)
  mg_tile$True_label = gsub(1, "POLE", mg_tile$True_label)
  mg_tile$True_label = gsub(0, "negative", mg_tile$True_label)
  mg_tile$Prediction = gsub(1, "POLE", mg_tile$Prediction)
  mg_tile$Prediction = gsub(0, "negative", mg_tile$Prediction)
  
  dir.create(file.path(paste("~/Documents/CPTAC-UCEC/Results/POLE_fusion/", arch1, "_", arch2, "_", arch3, sep="")), showWarnings = FALSE)
  write.csv(mg_slide, paste("~/Documents/CPTAC-UCEC/Results/POLE_fusion/", arch1, "_", arch2, "_", arch3, "/Test_slide.csv", sep=""))
  write.csv(mg_tile, paste("~/Documents/CPTAC-UCEC/Results/POLE_fusion/", arch1, "_", arch2, "_", arch3, "/Test_tile.csv", sep=""))
  
  return(list(mg_slide, mg_tile))
}


previous=read.csv("~/Documents/CPTAC-UCEC/Results/Statistics_POLE_fusion.csv")
existed=paste(previous$CNVL_Architecture, previous$CNVH_Architecture, previous$MSI_Architecture, sep='_')
OUTPUT = setNames(data.frame(matrix(ncol = 53, nrow = 0)), c("Feature", "CNVL_Architecture", "CNVH_Architecture", "MSI_Architecture", "Tiles", "Patient_ROC.95.CI_lower", "Patient_ROC",                 
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

arch_list = c('I1', 'I2', 'I3', 'I5', 'I6', 'X1', 'X2', 'X3', 'X4', 'F1', 'F2', 'F3', 'F4')
tiles = "NL6"
feature = "POLE"
pos = "POLE"
for (ta in arch_list){
  for (tb in arch_list){
    for (tc in arch_list){
      tryCatch(
        {
          if (paste(ta, tb, tc, sep="_") %in% existed){
            print("skip: ", ta, tb, tc)
            next
          } else{
            xxxx = generate(ta, tb, tc)
            Test_slide = xxxx[[1]]
            Test_tile = xxxx[[2]]
            
            # per patient level
            answers <- factor(Test_slide$True_label)
            results <- factor(Test_slide$Prediction)
            # statistical metrics
            CMP = confusionMatrix(data=results, reference=answers, positive = pos)
            # ROC
            roc =  roc(answers, Test_slide$POS_score, levels=c('negative', pos))
            rocdf = t(data.frame(ci.auc(roc)))
            colnames(rocdf) = c('ROC.95.CI_lower', 'ROC', 'ROC.95.CI_upper')
            # PRC
            SprcR = PRAUC(Test_slide$POS_score, factor(Test_slide$True_label))
            Sprls = list()
            for (j in 1:100){
              sampleddf = Test_slide[sample(nrow(Test_slide), round(nrow(Test_slide)*0.9)),]
              Sprc = PRAUC(sampleddf$POS_score, factor(sampleddf$True_label))
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
            Troc =  roc(Tanswers, Test_tile$POS_score, levels=c('negative', pos))
            Trocdf = t(data.frame(ci.auc(Troc)))
            colnames(Trocdf) = c('ROC.95.CI_lower', 'ROC', 'ROC.95.CI_upper')
            # PRC
            prcR = PRAUC(Test_tile$POS_score, factor(Test_tile$True_label))
            prls = list()
            for (j in 1:10){
              sampleddf = Test_tile[sample(nrow(Test_tile), round(nrow(Test_tile)*0.9)),]
              prc = PRAUC(sampleddf$POS_score, factor(sampleddf$True_label))
              prls[j] = prc
            }
            Tprcci = ci(as.numeric(prls))
            Tprcdf = data.frame('PRC.95.CI_lower' = Tprcci[2], 'PRC' = prcR, 'PRC.95.CI_upper' = Tprcci[3])
            # Combine and add prefix
            Toverall = cbind(Trocdf, Tprcdf, data.frame(t(CMT$overall)), data.frame(t(CMT$byClass)))
            colnames(Toverall) = paste('Tile', colnames(Toverall), sep='_')
            # Key names
            keydf = data.frame("Feature" = feature, "CNVL_Architecture" = ta, "CNVH_Architecture" = tb, "MSI_Architecture" = tc,  "Tiles" = tiles, "Positive" = pos)
            # combine all df and reset row name
            tempdf = cbind(keydf, soverall, Toverall)
            rownames(tempdf) <- NULL
            OUTPUT = rbind(OUTPUT, tempdf)
          }
        },
        error = function(error_message){
          message(error_message)
          message(ta, tb, tc)
          return(NA)
        }
      )
    }
  }
}

# Bind old with new; sort; save
New_OUTPUT = rbind(previous, OUTPUT)
New_OUTPUT = New_OUTPUT[order(-New_OUTPUT$Patient_ROC, -New_OUTPUT$Tile_ROC),]
write.csv(New_OUTPUT, file = "~/documents/CPTAC-UCEC/Results/Statistics_special.csv", row.names=FALSE)




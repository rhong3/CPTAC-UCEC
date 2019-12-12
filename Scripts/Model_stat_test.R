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
        testa = roc.test(roca, rocb, method="delong")
        
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
        testb = roc.test(roca, rocb, method="delong")
        
        TI_test[ara, arb] = round(testb$p.value, digits = 5)

      }
    }
  }
  write.csv(PA_test, file = paste("~/documents/CPTAC-UCEC/Results/AUROC_test/lite_",f ,"_patient_AUROC_test.csv", sep=''), row.names=TRUE)
  write.csv(TI_test, file = paste("~/documents/CPTAC-UCEC/Results/AUROC_test/lite_",f ,"_tile_AUROC_test.csv", sep=''), row.names=TRUE)
}
 
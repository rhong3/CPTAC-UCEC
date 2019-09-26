library(Hmisc)
library(Rfast)
library(dplyr)
# Correlation check
dict=read.csv('~/Documents/CPTAC-UCEC/dummy_His_MUT_joined.csv')
dict$subtype_Endometrioid = dict$subtype_Endometrioid-dict$subtype_0NA
dict$subtype_Endometrioid = gsub(-1, NA, dict$subtype_Endometrioid)
dict$subtype_MSI = dict$subtype_MSI-dict$subtype_0NA
dict$subtype_MSI = gsub(-1, NA, dict$subtype_MSI)
dict$subtype_POLE = dict$subtype_POLE-dict$subtype_0NA
dict$subtype_POLE = gsub(-1, NA, dict$subtype_POLE)
dict$subtype_Serous.like = dict$subtype_Serous.like-dict$subtype_0NA
dict$subtype_Serous.like = gsub(-1, NA, dict$subtype_Serous.like)
dict$MSIst_MSI.H = dict$MSIst_MSI.H-dict$MSIst_0NA
dict$MSIst_MSI.H = gsub(-1, NA, dict$MSIst_MSI.H)
dict$MSIst_MSS = dict$MSIst_MSS-dict$MSIst_0NA
dict$MSIst_MSS = gsub(-1, NA, dict$MSIst_MSS)
dict = subset(dict, select = -c(name, subtype_0NA, histology_Clear.cell, MSIst_MSI.L, MSIst_0NA))

# Yule's Y Coefficient of colligation column-wise calculation 
YuleY = function(col1, col2){
  newdf=data.frame(cbind(col1, col2))
  colnames(newdf) = c('A', 'B')
  newdf=na.omit(newdf)
  mtx = matrix(0, nrow=2, ncol=2)
  mtx[1,1] = nrow(newdf[which(newdf$A == 0 & newdf$B == 0),])
  mtx[1,2] = nrow(newdf[which(newdf$A == 0 & newdf$B == 1),])
  mtx[2,1] = nrow(newdf[which(newdf$A == 1 & newdf$B == 0),])
  mtx[2,2] = nrow(newdf[which(newdf$A == 1 & newdf$B == 1),])
  return(yule(mtx))
}

OUTPUT = setNames(data.frame((matrix(ncol = ncol(dict), nrow = ncol(dict)))), colnames(dict))
row.names(OUTPUT) = colnames(OUTPUT)
for (i in 1:ncol(dict)){
  for (j in 1:ncol(dict)){
      y = YuleY(dict[,i], dict[,j])
      OUTPUT[colnames(dict)[i], colnames(dict)[j]] = y
  }
}

OUTPUT[is.na(OUTPUT)] = 1
OUTPUT = round(OUTPUT, digits = 2)
write.csv(OUTPUT, '~/Documents/CPTAC-UCEC/YuleY_similarities.csv', row.names=TRUE)




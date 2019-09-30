### Summarize input at patient level
alldict = read.csv('~/documents/CPTAC-UCEC/dummy_His_MUT_joined.csv')
alldict$subtype_Endometrioid = alldict$subtype_Endometrioid-alldict$subtype_0NA
alldict$subtype_Endometrioid = as.numeric(gsub(-1, NA, alldict$subtype_Endometrioid))
alldict$subtype_MSI = alldict$subtype_MSI-alldict$subtype_0NA
alldict$subtype_MSI = as.numeric(gsub(-1, NA, alldict$subtype_MSI))
alldict$subtype_POLE = alldict$subtype_POLE-alldict$subtype_0NA
alldict$subtype_POLE = as.numeric(gsub(-1, NA, alldict$subtype_POLE))
alldict$subtype_Serous.like = alldict$subtype_Serous.like-alldict$subtype_0NA
alldict$subtype_Serous.like = as.numeric(gsub(-1, NA, alldict$subtype_Serous.like))
alldict$MSIst_MSI.H = alldict$MSIst_MSI.H-alldict$MSIst_0NA
alldict$MSIst_MSI.H = as.numeric(gsub(-1, NA, alldict$MSIst_MSI.H))
alldict$MSIst_MSS = alldict$MSIst_MSS-alldict$MSIst_0NA
alldict$MSIst_MSS = as.numeric(gsub(-1, NA, alldict$MSIst_MSS))
alldict = subset(alldict, select = -c(subtype_0NA, histology_Clear.cell, MSIst_0NA, MSIst_MSI.L))
OUTPUT = setNames(data.frame(matrix(ncol = 30, nrow = 6)), colnames(alldict)[2:31])
row.names(OUTPUT)=c('POS', 'NEG', 'POS.Rate', 'TCGA', 'CPTAC', 'Total')

# count numbers of patients
for (x in colnames(OUTPUT)){
  total = length(na.omit(alldict[,x]))
  if (length(alldict[,x]) ==  total){
    TCGA = length(grep('TCGA', alldict$name))
    CPTAC = length(grep('TCGA', alldict$name,invert = T))
  }
  else{
    TCGA = length(grep('TCGA', alldict[-which(is.na(alldict[,x])=='TRUE'),]$name))
    CPTAC = length(grep('TCGA', alldict[-which(is.na(alldict[,x])=='TRUE'),]$name,invert = T))
  }
  OUTPUT[1,x] = sum(na.omit(alldict[,x]))
  OUTPUT[2,x] = total-sum(na.omit(alldict[,x]))
  OUTPUT[3,x] = round(sum(na.omit(alldict[,x]))/total, 3)
  OUTPUT[4,x] = TCGA
  OUTPUT[5,x] = CPTAC
  OUTPUT[6,x] = total
}

write.csv(OUTPUT, '~/documents/CPTAC-UCEC/Patient_summary.csv')


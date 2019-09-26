library(Hmisc)
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
cor_table = rcorr(as.matrix(dict), type="pearson")
write.csv(cor_table$r, '~/Documents/CPTAC-UCEC/Correlation_coef.csv')
write.csv(cor_table$P, '~/Documents/CPTAC-UCEC/Correlation_P_value.csv')
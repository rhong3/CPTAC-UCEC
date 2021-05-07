# Summary to report
mut = c('ARID1A', 'ATM', 'BRCA2', 'CTCF', 'CTNNB1', 'FAT1', 'FBXW7', 'FGFR2', 'JAK1', 
        'KRAS', 'MTOR', 'PIK3CA', 'PIK3R1', 'PPP2R1A', 'PTEN', 'RPL22', 'TP53', 'ZFHX3')
mutation=read.csv("~/Documents/CPTAC-UCEC/Results/Statistics_mutations.csv")
colnames(mutation) = gsub('Mutation', 'Feature', colnames(mutation))
mutation = mutation[mutation$Feature %in% mut, ]
histology = read.csv("~/Documents/CPTAC-UCEC/Results/Statistics_histology.csv")
histology$Feature = 'Histology'
special = read.csv("~/Documents/CPTAC-UCEC/Results/Statistics_special.csv")
special = special[, -4]
special$Feature = gsub('SL', 'CNVH in Endometrioid', special$Feature)
MSI =read.csv("~/Documents/CPTAC-UCEC/Results/Statistics_MSI.csv")
MSI$Feature = 'MSI-High'

merged = rbind(mutation, histology, special, MSI)
merged$Architecture = gsub('I1', 'InceptionV1', merged$Architecture)
merged$Architecture = gsub('I2', 'InceptionV2', merged$Architecture)
merged$Architecture = gsub('I3', 'InceptionV3', merged$Architecture)
merged$Architecture = gsub('I4', 'InceptionV4', merged$Architecture)
merged$Architecture = gsub('I5', 'InceptionResnetV1', merged$Architecture)
merged$Architecture = gsub('I6', 'InceptionResnetV2', merged$Architecture)
merged$Architecture = gsub('X1', 'Panoptes2', merged$Architecture)
merged$Architecture = gsub('X2', 'Panoptes1', merged$Architecture)
merged$Architecture = gsub('X3', 'Panoptes4', merged$Architecture)
merged$Architecture = gsub('X4', 'Panoptes3', merged$Architecture)
merged$Architecture = gsub('F1', 'Panoptes2 with clinical', merged$Architecture)
merged$Architecture = gsub('F2', 'Panoptes1 with clinical', merged$Architecture)
merged$Architecture = gsub('F3', 'Panoptes4 with clinical', merged$Architecture)
merged$Architecture = gsub('F4', 'Panoptes3 with clinical', merged$Architecture)

is.num <- sapply(merged, is.numeric)
merged[is.num] <- lapply(merged[is.num], round, 3)
merged$Patient_AUROC = paste(merged$Patient_ROC, ' (', merged$Patient_ROC.95.CI_lower, '-', merged$Patient_ROC.95.CI_upper, ')', sep='')
merged$Patient_AUPR = paste(merged$Patient_PRC, ' (', merged$Patient_PRC.95.CI_lower, '-', merged$Patient_PRC.95.CI_upper, ')', sep='')
merged$Patient_Accuracy = paste(merged$Patient_Accuracy, ' (', merged$Patient_AccuracyLower, '-', merged$Patient_AccuracyUpper, ')', sep='')
merged$Tile_AUROC = paste(merged$Tile_ROC, ' (', merged$Tile_ROC.95.CI_lower, '-', merged$Tile_ROC.95.CI_upper, ')', sep='')
merged$Tile_AUPR = paste(merged$Tile_PRC, ' (', merged$Tile_PRC.95.CI_lower, '-', merged$Tile_PRC.95.CI_upper, ')', sep='')
merged$Tile_Accuracy = paste(merged$Tile_Accuracy, ' (', merged$Tile_AccuracyLower, '-', merged$Tile_AccuracyUpper, ')', sep='')

merged = merged[order(merged[,1], -merged[,5], -merged[,28]),]
merged = cbind(merged[, c(1:3, 52:53, 10, 54:55, 34)], merged[, c(11, 14:27, 35, 38:51)])



# NL5 mixed 
mixed = merged[merged$Tiles %in% c('NL5') & merged$Architecture %in% c('InceptionV1', 'InceptionV2', 'InceptionV3', 'InceptionResnetV1', 'InceptionResnetV2', 
                                                                       'Panoptes1', 'Panoptes2', 'Panoptes3', 'Panoptes4', 
                                                                       'Panoptes1 with clinical' ,'Panoptes2 with clinical', 'Panoptes3 with clinical', 'Panoptes4 with clinical'), ]
mixed$Resolution = 'multi'
# mixed[mixed$Tiles == 'NL8', ]$Resolution = '5X'
# mixed[mixed$Tiles == 'NL9', ]$Resolution = '2.5X'
mixed[mixed$Tiles == 'NL5' & mixed$Architecture %in% c('InceptionV1', 'InceptionV2', 'InceptionV3', 'InceptionResnetV1', 'InceptionResnetV2'), ]$Resolution = '10X'
mixed = mixed[, c(1:2, 40, 4:39)]

write.csv(mixed, '~/Documents/CPTAC-UCEC/Results/Mixed_models_summary.csv', row.names=FALSE)

# NL6 independent
idp = merged[merged$Tiles %in% c('NL6', 'NLI') & merged$Architecture %in% c('InceptionV1', 'InceptionV2', 'InceptionV3', 'InceptionResnetV1', 'InceptionResnetV2', 
                                                                       'Panoptes1', 'Panoptes2', 'Panoptes3', 'Panoptes4', 
                                                                       'Panoptes1 with clinical' ,'Panoptes2 with clinical', 'Panoptes3 with clinical', 'Panoptes4 with clinical'), ]

idp$Resolution = 'multi'
idp[idp$Architecture %in% c('InceptionV1', 'InceptionV2', 'InceptionV3', 'InceptionResnetV1', 'InceptionResnetV2'), ]$Resolution = '10X'
idp= idp[, c(1:2, 40, 4:39)]

write.csv(idp, '~/Documents/CPTAC-UCEC/Results/Independent_models_summary.csv', row.names=FALSE)


# NYU Summary
NYU = read.csv("~/Documents/CPTAC-UCEC/Results/Statistics_NYU.csv")
NYU$Architecture = gsub('I1', 'InceptionV1', NYU$Architecture)
NYU$Architecture = gsub('I2', 'InceptionV2', NYU$Architecture)
NYU$Architecture = gsub('I3', 'InceptionV3', NYU$Architecture)
NYU$Architecture = gsub('I4', 'InceptionV4', NYU$Architecture)
NYU$Architecture = gsub('I5', 'InceptionResnetV1', NYU$Architecture)
NYU$Architecture = gsub('I6', 'InceptionResnetV2', NYU$Architecture)
NYU$Architecture = gsub('X1', 'Panoptes2', NYU$Architecture)
NYU$Architecture = gsub('X2', 'Panoptes1', NYU$Architecture)
NYU$Architecture = gsub('X3', 'Panoptes4', NYU$Architecture)
NYU$Architecture = gsub('X4', 'Panoptes3', NYU$Architecture)
NYU$Architecture = gsub('F1', 'Panoptes2 with clinical', NYU$Architecture)
NYU$Architecture = gsub('F2', 'Panoptes1 with clinical', NYU$Architecture)
NYU$Architecture = gsub('F3', 'Panoptes4 with clinical', NYU$Architecture)
NYU$Architecture = gsub('F4', 'Panoptes3 with clinical', NYU$Architecture)
colnames(NYU) = gsub('Tiles', 'Split', colnames(NYU))
NYU$Split = gsub('NL5', 'Mixed', NYU$Split)
NYU$Split = gsub('NL6', 'Independent', NYU$Split)
NYU$Feature = gsub('histology', 'Histology', NYU$Feature)
NYU$Feature = gsub('MSI', 'MSI-High', NYU$Feature)
NYU$Feature = gsub('SL', 'CNVH in Endometrioid', NYU$Feature)

is.num <- sapply(NYU, is.numeric)
NYU[is.num] <- lapply(NYU[is.num], round, 3)
NYU$Patient_AUROC = paste(NYU$Patient_ROC, ' (', NYU$Patient_ROC.95.CI_lower, '-', NYU$Patient_ROC.95.CI_upper, ')', sep='')
NYU$Patient_AUPR = paste(NYU$Patient_PRC, ' (', NYU$Patient_PRC.95.CI_lower, '-', NYU$Patient_PRC.95.CI_upper, ')', sep='')
NYU$Patient_Accuracy = paste(NYU$Patient_Accuracy, ' (', NYU$Patient_AccuracyLower, '-', NYU$Patient_AccuracyUpper, ')', sep='')
NYU$Tile_AUROC = paste(NYU$Tile_ROC, ' (', NYU$Tile_ROC.95.CI_lower, '-', NYU$Tile_ROC.95.CI_upper, ')', sep='')
NYU$Tile_AUPR = paste(NYU$Tile_PRC, ' (', NYU$Tile_PRC.95.CI_lower, '-', NYU$Tile_PRC.95.CI_upper, ')', sep='')
NYU$Tile_Accuracy = paste(NYU$Tile_Accuracy, ' (', NYU$Tile_AccuracyLower, '-', NYU$Tile_AccuracyUpper, ')', sep='')

NYU = NYU[order(NYU[,3], NYU[,1], -NYU[,5], -NYU[,29]),]
NYU = cbind(NYU[, c(1:3, 52:53, 10, 54:55, 34)], NYU[, c(11, 14:27, 35, 38:51)])

NYU = NYU[NYU$Architecture %in% c('InceptionV1', 'InceptionV2', 'InceptionV3', 'InceptionResnetV1', 'InceptionResnetV2', 
                                                                       'Panoptes1', 'Panoptes2', 'Panoptes3', 'Panoptes4', 
                                                                       'Panoptes1 with clinical' ,'Panoptes2 with clinical', 'Panoptes3 with clinical', 'Panoptes4 with clinical'), ]
NYU$Resolution = 'multi'
NYU[NYU$Architecture %in% c('InceptionV1', 'InceptionV2', 'InceptionV3', 'InceptionResnetV1', 'InceptionResnetV2'), ]$Resolution = '10X'
NYU = NYU[, c(1:3, 40, 4:39)]

write.csv(NYU, '~/Documents/CPTAC-UCEC/Results/NYU_models_summary.csv', row.names=FALSE)




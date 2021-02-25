# NYU data summary
library(ComplexHeatmap)
library(dplyr)
library(fastDummies)
library(tidyr)
library(circlize)

summ = read.csv('~/documents/CPTAC-UCEC/NYU/sum.csv')
summ = summ[, -c(2,6)]
summ.m = summ %>%
  group_by(Patient_ID) %>%
  summarize(n=n())
colnames(summ.m) = c("Patient_ID", "H&E_slides")
summ.x = unique(summ)
label = read.csv('~/documents/CPTAC-UCEC/NYU/label.csv')
label = label[, c(1, 7)]
colnames(label) = c('Patient_ID', 'P53_overexpression')
summ_label = inner_join(summ.x, label, by=c('Patient_ID'))
summ_label['histology'] = tolower(summ_label$histology)
summ_label['histology'] = gsub('mixed serous/endometrioid', 'mixed', summ_label$histology)
summ_label['histology'] = gsub('mixed endometrioid/clear cell', 'mixed', summ_label$histology)
summ_label = inner_join(summ_label, summ.m, by=c('Patient_ID'))

IHC = read.csv('~/documents/CPTAC-UCEC/NYU/IHC_sum.csv')
IHC = IHC[, c(1,2)]
IHC = cbind(IHC, dummy_cols(IHC$Slide_ID))
IHC = IHC[, -c(2,3)]
colnames(IHC) = gsub(".data", "IHC", colnames(IHC))
FFF = IHC %>%
  group_by(Patient_ID) %>%
  summarise_all(mean)
FFF[, 2:6] = as.integer(FFF[, 2:6]>0)
summ_label_ihc = left_join(summ_label, FFF, by=c('Patient_ID'))
summ_label_ihc[is.na(summ_label_ihc)] = 0
summ_label_ihc[summ_label_ihc['FIGO'] == '', 4] = NA
summ_label_ihc['FIGO'] = gsub('FIGO ', '', summ_label_ihc$FIGO)
summ_label_ihc = summ_label_ihc[order(summ_label_ihc$subtype, summ_label_ihc$histology), ]

write.csv(summ_label_ihc, '~/documents/CPTAC-UCEC/NYU/NYU_data_summary.csv', row.names = FALSE)

# Heatmap
summ_label_ihc = read.csv('~/documents/CPTAC-UCEC/NYU/NYU_data_summary.csv')
binaries = c('gray90','gray10')

get_color = function(colors,factor){
  levels=levels(factor)
  print(levels)
  res = colors[unique(as.numeric(sort(factor)))]
  res = res[!is.na(res)]
  names(res) = levels
  print(res)
  return(res)
}

rownames(summ_label_ihc)=as.character(summ_label_ihc$Patient_ID)

## get binary colors
summ_label_ihc[,7:11] = lapply(summ_label_ihc[,7:11],
                               function(x)as.factor(x))
ColSide=lapply(summ_label_ihc[,7:11],
               function(x)get_color(binaries,x))

ColSide[['subtype']]=get_color(colors=c('#7fc97f','#beaed4','#fdc086'),
                               factor=summ_label_ihc$subtype)

ColSide[['histology']]=get_color(colors=c('#1b9e77','#d95f02', '#7570b3', '#e7298a','#66a61e'),
                                 factor = summ_label_ihc$histology)

ColSide[['FIGO']]=colorRamp2(breaks=range(summ_label_ihc$FIGO, na.rm=T), 
                            colors=c('#fee5d9', '#fb6a4a'))

ColSide[['P53_overexpression']]=colorRamp2(breaks=range(summ_label_ihc$P53_overexpression, na.rm=T),
                                           colors=c("#eff3ff","#2171b5"))

ColSide[['H.E_slides']]=colorRamp2(breaks=range(summ_label_ihc$H.E_slides, na.rm=T),
                                   colors=c("#fcfbfd", "#3f007d"))


ca = HeatmapAnnotation(df = summ_label_ihc[order(summ_label_ihc$histology,summ_label_ihc$subtype),-1], na_col ='white',
                       which = 'column',
                       annotation_name_gp = gpar(fontsize =20,fontface='bold'),
                       annotation_height = unit(rep(0.5,length(ColSide)), "inch"),
                       border = F,
                       gap = unit(rep(0,length(ColSide)), "inch"),
                       annotation_legend_param = list(title_gp = gpar(fontsize = 22,fontface = 'bold'),
                                                      labels_gp = gpar(fontsize = 20),
                                                      direction='horizontal',
                                                      #nrow =2, ncol=10,
                                                      grid_width= unit(0.3,'inch'),
                                                      grid_height = unit(0.3,'inch')
                       ),
                       col = ColSide,
                       show_annotation_name =T)

ph = matrix(NA ,ncol=446,nrow=0)

plot_heatmap=Heatmap(ph[,order(summ_label_ihc$histology,summ_label_ihc$subtype)], 
                     top_annotation = ca, cluster_columns = F, cluster_rows = F, show_heatmap_legend = F)


out_dir = '~/documents/CPTAC-UCEC/NYU/'
pdf(file = paste(out_dir,'NYU_data_summary.pdf',sep='/'),
    width =15, height = 5, bg='white')
draw(plot_heatmap, annotation_legend_side = "bottom")
graphics.off()


# alignment evaluation
library(ggplot2)
align = read.csv('~/documents/GYN-HE-IHC/align/align_eval.csv')
pdf(file = paste('~/documents/GYN-HE-IHC/align/align_eval.pdf',sep='/'),
    width =4, height = 4, bg='white')
ggplot(align, aes(Align_score)) +
  geom_histogram() + 
  labs(x='Align Score (median=0.91776)') +
  geom_vline(aes(xintercept = median(Align_score)),col='red',size=0.5)
dev.off()







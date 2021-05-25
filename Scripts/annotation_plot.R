dat = read.table(file='/media/lwk/data/ucec_image/Fusion_dummy_His_MUT_joined.csv',
                 header=T, sep=',')
colnames(dat)
dat = subset(dat,(histology_Endometrioid==1|histology_Serous==1)&histology_Mixed==0)
dat = dat[,c(1:31,33:38)]
subtype=apply(dat[,26:30],1,function(x)which(x!=0))
subtype = as.factor(subtype)
levels(subtype)=c(NA,'Endometrioid','MSI','POLE','Serous_like')

histology=apply(dat[,31:32],1,function(x)which(x!=0))
histology=as.factor(histology)
levels(histology)=c('Endometrioid','serous')

MSI = apply(dat[,33:35],1,function(x)which(x!=0))
MSI = as.factor(MSI)
levels(MSI)=c(NA,'MSI_H','MSI_S')

plot_dat = cbind(dat[,2:25],subtype, histology, MSI, 
                 dat[,36:37])


for (i in 1:24){
  plot_dat[,i]=as.factor(plot_dat[,i])
}

plot_dat$age=as.numeric(plot_dat$age)
library(ComplexHeatmap)
library(circlize)

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

rownames(plot_dat)=as.character(dat$name)

## get binary colors
ColSide=lapply(plot_dat[,1:24],
               function(x)get_color(binaries,x))

ColSide[['subtype']]=get_color(colors=c('#7fc97f','#beaed4','#fdc086','#f0027f'),
                               factor=plot_dat$subtype)

ColSide[['histology']]=get_color(colors=c('#66c2a5','#fc8d62'),
                                 factor = plot_dat$histology)

ColSide[['MSI']]=get_color(colors=c('#e78ac3','#8da0cb'),
                           factor = plot_dat$MSI)

ColSide[['BMI']]=colorRamp2(breaks=range(plot_dat$BMI,na.rm=T),
                            colors=c("#eff3ff","#2171b5"))
ColSide[['age']]=colorRamp2(breaks=range(plot_dat$age,na.rm=T),
                            colors=c("#fee5d9","#cb181d"))

#ColSide[['age']]=colorRampPalette(c("#fee5d9","#cb181d"))(256)


ca = HeatmapAnnotation(df = plot_dat[order(plot_dat$histology,plot_dat$subtype),], na_col ='white',
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
ph = matrix(rnorm(446*2),ncol=446,nrow=2)
colnames(ph)=rownames(plot_dat)

plot_heatmap=Heatmap(ph[,order(plot_dat$histology,plot_dat$subtype)], 
                     top_annotation = ca, cluster_columns = F, cluster_rows = F, show_heatmap_legend = F)

out_dir = '/media/lwk/data/ucec_image/'
pdf(file = paste(out_dir,'annotation.pdf',sep='/'),
    width = 45, height = 10, bg='white')
draw(plot_heatmap,annotation_legend_side = "bottom")
graphics.off()

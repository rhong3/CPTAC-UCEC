# Summary of Images
library(ggplot2)
library(tidyr)
library(tibble)
library(dplyr)
library(plyr)
slide_ct = read.csv('~/documents/CPTAC-UCEC/Results/slide_dim.csv')
patient_ct = read.csv('~/documents/CPTAC-UCEC/Patient_summary.csv', row.names = 1)
patient_ct = data.frame(t(patient_ct[c(1,2, 6), ]))
patient_ct<- patient_ct %>% rownames_to_column("Feature")
ppos = patient_ct[,c(1,2)]
ppos['label'] = "Positive"
colnames(ppos) = c('Feature', 'Count', 'Label')
pneg = patient_ct[,c(1,3)]
pneg['label'] = "Negative"
colnames(pneg) = c('Feature', 'Count', 'Label')
patient = rbind(pneg, ppos)
sss =c('ARID5B', 'EGFR', 'ERBB2', 'FAT4', 'MLH1', 'MSIst_MSS', 'PIK3R2', 'MSIst_MSI.H', 'histology_Mixed')
patient = patient[!(patient$Feature %in% sss), ]
patient$Feature = gsub('histology_','', patient$Feature)
patient$Feature = gsub('subtype_Endometrioid','CNV-L', patient$Feature)
patient$Feature = gsub('subtype_','', patient$Feature)
patient$Feature = gsub('Serous.like','CNV-H', patient$Feature)



spp = read.csv('~/documents/CPTAC-UCEC/patient_slides_count.csv')

pdf(file=paste("~/documents/CPTAC-UCEC/Results/slides_per_patient.pdf", sep=''),
    width=6.5,height=5)

pie = data.frame(group=c('1 slide','2 slides','3 slides'), value=c(length(spp[spp$num_of_slides==1,]$patient),length(spp[spp$num_of_slides==2,]$patient),length(spp[spp$num_of_slides==3,]$patient)))
p = ggplot(pie, aes(x="", y=value, fill=group)) +
  geom_bar(stat="identity", width=1) + scale_fill_manual(values=c("#55DDE0", "#33658A", "#999999"))  +
  coord_polar("y", start=0) + labs(x = NULL, y = NULL, fill = NULL)+ theme_classic() + theme(axis.line = element_blank(),
                                                                                                                           axis.text = element_blank(),
                                                                                                                           axis.ticks = element_blank()
                                                                                                                           )

grid.arrange(p, nrow=1, ncol=1)
dev.off()

pdf(file=paste("~/documents/CPTAC-UCEC/Results/slides_dimension.pdf", sep=''),
    width=5,height=3)
pp_DF = slide_ct[, c(5,6)]
pp = pp_DF %>% 
  gather(key=Length, value=Pixel) %>% 
  ggplot(aes(x=Pixel,fill=Length)) + 
  geom_histogram(position="dodge", binwidth=5000)+ scale_fill_grey() +theme_bw() + theme(panel.border = element_blank(), panel.grid.major = element_blank(),
                                                                        panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))+
  font("xlab", size = 0, color = "black")+
  font("ylab", size = 0, color = "black", face = "bold")+
  font("xy.text", size = 12, color = "black", face = "bold")
grid.arrange(pp, nrow=1, ncol=1)
dev.off()

pdf(file=paste("~/documents/CPTAC-UCEC/Results/X20Xtile_count.pdf", sep=''),
    width=5,height=3)
p<-ggplot(slide_ct, aes(x=X20Xtiles)) + geom_histogram(binwidth=500)+theme_bw() + theme(panel.border = element_blank(), panel.grid.major = element_blank(),
                                                                                                   panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))+
  font("xlab", size = 0, color = "black")+
  font("ylab", size = 0, color = "black", face = "bold")+
  font("xy.text", size = 12, color = "black", face = "bold")
grid.arrange(p, nrow=1, ncol=1)
dev.off()

pdf(file=paste("~/documents/CPTAC-UCEC/Results/X10Xtile_count.pdf", sep=''),
    width=5,height=3)
p<-ggplot(slide_ct, aes(x=X10Xtiles)) + geom_histogram(binwidth=100)+theme_bw() + theme(panel.border = element_blank(), panel.grid.major = element_blank(),
                                                                                        panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))+
  font("xlab", size = 0, color = "black")+
  font("ylab", size = 0, color = "black", face = "bold")+
  font("xy.text", size = 12, color = "black", face = "bold")
grid.arrange(p, nrow=1, ncol=1)
dev.off()

pdf(file=paste("~/documents/CPTAC-UCEC/Results/X5Xtile_count.pdf", sep=''),
    width=5,height=3)
p<-ggplot(slide_ct, aes(x=X5Xtiles)) + geom_histogram(binwidth=25)+theme_bw() + theme(panel.border = element_blank(), panel.grid.major = element_blank(),
                                                                                        panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))+
  font("xlab", size = 0, color = "black")+
  font("ylab", size = 0, color = "black", face = "bold")+
  font("xy.text", size = 12, color = "black", face = "bold")
grid.arrange(p, nrow=1, ncol=1)
dev.off()

pdf(file=paste("~/documents/CPTAC-UCEC/Results/X2.5Xtile_count.pdf", sep=''),
    width=5,height=3)
p<-ggplot(slide_ct, aes(x=X2.5Xtiles)) + geom_histogram(binwidth=5)+theme_bw() + theme(panel.border = element_blank(), panel.grid.major = element_blank(),
                                                                                      panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))+
  font("xlab", size = 0, color = "black")+
  font("ylab", size = 0, color = "black", face = "bold")+
  font("xy.text", size = 12, color = "black", face = "bold")
grid.arrange(p, nrow=1, ncol=1)
dev.off()

pdf(file=paste("~/documents/CPTAC-UCEC/Results/Patient_count.pdf", sep=''),
    width=10,height=6)
p = ggplot(data=patient, aes(x = reorder(Feature, -Count), y=Count, fill=Label)) +
  geom_bar(stat="identity")+ scale_fill_brewer(palette="Greys") +theme_bw() + theme(panel.border = element_blank(), panel.grid.major = element_blank(), legend.text=element_text(size=12),
                                     panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"), axis.text.x = element_text(angle = 90, hjust = 1))+
  font("xlab", size = 0, color = "black")+
  font("ylab", size = 0, color = "black", face = "bold")+
  font("xy.text", size = 12, color = "black", face = "bold")
grid.arrange(p, nrow=1, ncol=1)
dev.off()



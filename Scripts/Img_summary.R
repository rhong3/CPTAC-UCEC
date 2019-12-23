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
ppos['label'] = "POS"
colnames(ppos) = c('Feature', 'Count', 'Label')
pneg = patient_ct[,c(1,3)]
pneg['label'] = "NEG"
colnames(pneg) = c('Feature', 'Count', 'Label')
patient = rbind(pneg, ppos)
sss =c('ARID5B', 'EGFR', 'ERBB2', 'FAT4', 'MLH1', 'MSIst_MSS', 'PIK3R2')
patient = patient[!(patient$Feature %in% sss), ]

spp = read.csv('~/documents/CPTAC-UCEC/patient_slides_count.csv')

pdf(file=paste("~/documents/CPTAC-UCEC/Results/slides_per_patient.pdf", sep=''),
    width=5,height=5)
p<-ggplot(spp, aes(x=num_of_slides)) + geom_histogram(color='grey', binwidth=1)+theme_bw() + theme(panel.border = element_blank(), panel.grid.major = element_blank(),
                                                                                                   panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))
grid.arrange(p, nrow=1, ncol=1)
dev.off()

pdf(file=paste("~/documents/CPTAC-UCEC/Results/slides_dimension.pdf", sep=''),
    width=10,height=10)
pp_DF = slide_ct[, c(5,6)]
pp = pp_DF %>% 
  gather(key=Length, value=Pixel) %>% 
  ggplot(aes(x=Pixel,fill=Length)) + 
  geom_histogram(position="dodge", binwidth=5000)+ scale_fill_grey() +theme_bw() + theme(panel.border = element_blank(), panel.grid.major = element_blank(),
                                                                        panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))
grid.arrange(pp, nrow=1, ncol=1)
dev.off()

pdf(file=paste("~/documents/CPTAC-UCEC/Results/X20Xtile_count.pdf", sep=''),
    width=15,height=10)
p<-ggplot(slide_ct, aes(x=X20Xtiles)) + geom_histogram(binwidth=500)+theme_bw() + theme(panel.border = element_blank(), panel.grid.major = element_blank(),
                                                                                                   panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))
grid.arrange(p, nrow=1, ncol=1)
dev.off()

pdf(file=paste("~/documents/CPTAC-UCEC/Results/X10Xtile_count.pdf", sep=''),
    width=15,height=10)
p<-ggplot(slide_ct, aes(x=X10Xtiles)) + geom_histogram(binwidth=100)+theme_bw() + theme(panel.border = element_blank(), panel.grid.major = element_blank(),
                                                                                        panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))
grid.arrange(p, nrow=1, ncol=1)
dev.off()

pdf(file=paste("~/documents/CPTAC-UCEC/Results/X5Xtile_count.pdf", sep=''),
    width=15,height=10)
p<-ggplot(slide_ct, aes(x=X5Xtiles)) + geom_histogram(binwidth=25)+theme_bw() + theme(panel.border = element_blank(), panel.grid.major = element_blank(),
                                                                                        panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))
grid.arrange(p, nrow=1, ncol=1)
dev.off()

pdf(file=paste("~/documents/CPTAC-UCEC/Results/X2.5Xtile_count.pdf", sep=''),
    width=15,height=10)
p<-ggplot(slide_ct, aes(x=X2.5Xtiles)) + geom_histogram(binwidth=5)+theme_bw() + theme(panel.border = element_blank(), panel.grid.major = element_blank(),
                                                                                      panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))
grid.arrange(p, nrow=1, ncol=1)
dev.off()

pdf(file=paste("~/documents/CPTAC-UCEC/Results/Patient_count.pdf", sep=''),
    width=15,height=10)
p = ggplot(data=patient, aes(x=Feature, y=Count, fill=Label)) +
  geom_bar(stat="identity")+ scale_fill_brewer(palette="Greys") +theme_bw() + theme(panel.border = element_blank(), panel.grid.major = element_blank(),
                                     panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"), axis.text.x = element_text(angle = 45, hjust = 1))
grid.arrange(p, nrow=1, ncol=1)
dev.off()


# Scatter plot of logits
library(ggplot2)
library(dplyr)

NYU = read.csv("NYU/NYU_data_summary.csv")
NYU = NYU[order(NYU$Patient_ID),]
for (i in c('NL5', 'NL6')){
  for (j in c("I1", "I2", "I3", "I5", "I6", "X1", "X2", "X3", "X4", "F1", "F2", "F3", "F4")){
    Slogit = read.csv(paste("Results/NYU_test_revision/", j, "his_NYU_", i, "/out/Test_slide_logit.csv", sep=''))
    
    Slogit = cbind(NYU, Slogit)
    
    ggplot(Slogit, aes(logit0, logit1, color=histology)) +
      geom_point() + xlab("logit0")+ ylab("logit1")
    ggsave(paste("Results/NYU_test_revision/", j, "his_NYU_", i, "/out/slide_logits.png", sep=''), plot=last_plot(), width = 6, height = 4, units="in", dpi=300)
    
    Slogits_mean = Slogit %>%
      group_by(histology) %>%
      summarize(mean_logit0=mean(logit0), mean_logit1=mean(logit1), sd_logit0=sd(logit0), sd_logit1=sd(logit1), n=n())
    
    write.csv(Slogits_mean, paste("Results/NYU_test_revision/", j, "his_NYU_", i, "/out/mean_logit.csv", sep=''))
    
    ggplot(Slogits_mean, aes(mean_logit0, mean_logit1, color=histology)) +
      geom_point() + xlab("logit0 mean")+ ylab("logit1 mean")
    ggsave(paste("Results/NYU_test_revision/", j, "his_NYU_", i, "/out/slide_logits_mean.png", sep=''), plot=last_plot(), width = 6, height = 4, units="in", dpi=300)
    
    Tlogit = read.csv(paste("Results/NYU_test_revision/", j, "his_NYU_", i, "/out/Test_tile_logit.csv", sep=''))
    colnames(Tlogit) = gsub("slide", "Patient_ID", colnames(Tlogit))
    Tlogit = left_join(Tlogit, NYU, by="Patient_ID")
    
    Tlogits_mean = Tlogit %>%
      group_by(histology) %>%
      summarize(mean_logit0=mean(logit0), mean_logit1=mean(logit1), sd_logit0=sd(logit0), sd_logit1=sd(logit1), n=n())
    
    ggplot(Tlogits_mean, aes(mean_logit0, mean_logit1, color=histology)) +
      geom_point() + xlab("logit0 mean")+ ylab("logit1 mean")
    ggsave(paste("Results/NYU_test_revision/", j, "his_NYU_", i, "/out/tile_logits.png", sep=''), plot=last_plot(), width = 6, height = 4, units="in", dpi=300)
    
    input_file=paste("Results/NYU_test_revision/", j, "his_NYU_", i, "/out/For_tSNE.csv", sep='')
    start=12
    bins=50
    POS_score=c('Serous_score')
    TLB = 1 # ST is 2, others 1
    MDP = 0.5 # 0.5 for binary; 1/length(POS_score)
    
    library(Rtsne)
    ori_dat = read.table(file=input_file,header=T,sep=',')
    
    sp_ori_dat=ori_dat[sample(nrow(ori_dat), 20000), ]
    sp_ori_dat = sp_ori_dat %>%
      group_by(slide) %>%
      summarise_all(mean)
    
    X = as.matrix(sp_ori_dat[,start:dim(sp_ori_dat)[2]])
    res = Rtsne(X, initial_dims=100, check_duplicates = FALSE, perplexity = 10)
    Y=res$Y
    out_dat = cbind(sp_ori_dat[,1:(start-1)],Y)
    
    dat = cbind(out_dat,x_bin=cut(out_dat[,start],bins),
                y_bin=cut(out_dat[,(start+1)],bins))
    
    dat = cbind(dat, x_int = as.numeric(dat$x_bin),
                y_int = as.numeric(dat$y_bin))
    
    colnames(dat)[start:(start+1)]=c('tsne1','tsne2')
    
    dat$True_label=as.factor(dat$True_label)
    dat$slide=as.factor(dat$slide)
    
    
    colnames(dat) = gsub("slide", "Patient_ID", colnames(dat))
    dat = left_join(dat, NYU, by="Patient_ID")
    
    library(ggplot2)
    library(gridExtra)
    
    ggplot(data=dat,aes_string(x='tsne1',y='tsne2'))+
      geom_point(aes(col=histology),alpha=0.8)+
      xlim(-20,20)+
      ylim(-20,20)+ 
      theme_bw() + theme(panel.border = element_blank(), panel.grid.major = element_blank(),
                         panel.grid.minor = element_blank(), 
                         axis.line = element_line(colour = "black"), legend.position='right')
    ggsave(paste("Results/NYU_test_revision/", j, "his_NYU_", i, "/out/slide_tSNE.png", sep=''), plot=last_plot(), width = 6, height = 4, units="in", dpi=300)
  }
}



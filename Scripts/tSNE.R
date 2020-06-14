## reduce dimensionality of the acitvation layer of model
## visualize the manifold

# args = commandArgs(trailingOnly=FALSE)
# input_file=args[1]
# output_file=args[2]
# out_fig=args[3]
# start=args[4]
# bins=args[5]
# POS_score=args[6]

# I START AT 9, X START AT 12; ST start I at 11, X at 14
inlist=c('X1CCNA2', 'X2CCNA2', 'X3CCNA2', 'F1CCNA2', 'F2CCNA2', 'F3CCNA2', 'X1CCND1', 'X3CCND1', 'X4CCND1', 'F3CCND1', 'F4CCND1', 'X2LINE1_ORF1p', 'F1NBN-S343')

for(xx in inlist){
  input_file=paste('~/documents/CPTAC-UCEC/Results/NL5/LINE1/',xx,'/out/For_tSNE.csv',sep='')
  output_file=paste('~/documents/CPTAC-UCEC/Results/NL5/LINE1/',xx,'/out/tSNE_P_N.csv',sep='')
  sampled_file=paste('~/documents/CPTAC-UCEC/Results/NL5/LINE1/',xx,'/out/tSNE_sampled.csv',sep='')
  out_fig=paste('~/documents/CPTAC-UCEC/Results/NL5/LINE1/',xx,'/out/P_N.pdf',sep='')
  start=12
  bins=50
  POS_score=c('POS_score')
  TLB = 1 # ST is 2, others 1
  MDP = 0.5 # 0.5 for binary; 1/length(POS_score)
  
  library(Rtsne)
  ori_dat = read.table(file=input_file,header=T,sep=',')
  # P = ori_dat[which(ori_dat$Prediction==1),]
  # N = ori_dat[which(ori_dat$Prediction==0),]
  # N = ori_dat[sample(nrow(N), 20000), ]
  # sp_ori_dat = rbind(P, N)
  # SAMPLE 20000 FOR LEVEL 1 & 2; NO SAMPLE FOR LEVEL 3
  # sp_ori_dat=ori_dat[sample(nrow(ori_dat), 20000), ]
  sp_ori_dat=ori_dat
  write.table(sp_ori_dat, file=sampled_file, row.names = F, sep=',')
  # sp_ori_dat=ori_dat
  X = as.matrix(sp_ori_dat[,start:dim(sp_ori_dat)[2]])
  res = Rtsne(X, initial_dims=100, check_duplicates = FALSE)
  Y=res$Y
  out_dat = cbind(sp_ori_dat[,1:(start-1)],Y)
  
  dat = cbind(out_dat,x_bin=cut(out_dat[,start],bins),
              y_bin=cut(out_dat[,(start+1)],bins))
  
  dat = cbind(dat, x_int = as.numeric(dat$x_bin),
              y_int = as.numeric(dat$y_bin))
  
  colnames(dat)[start:(start+1)]=c('tsne1','tsne2')
  
  dat$True_label=as.factor(dat$True_label)
  dat$slide=as.factor(dat$slide)
  
  write.table(dat, file=output_file, row.names = F, sep=',')
  
  
  ## plot the manifold with probability
  library(ggplot2)
  library(gridExtra)
  palist <- list()
  pblist <- list()
  for(i in 1:length(POS_score)){
    palist[[i]]=ggplot(data=dat,aes_string(x='tsne1',y='tsne2',col=POS_score[i]))+
      scale_color_gradient2(high='red',mid='gray',low='steelblue',midpoint=MDP)+
      geom_point(alpha=1, size=1)+ scale_shape(solid = TRUE)+
      xlim(-60,60)+
      ylim(-60,60)+
      theme_bw() + theme(panel.border = element_blank(), panel.grid.major = element_blank(),
                         panel.grid.minor = element_blank(), 
                         axis.line = element_line(colour = "black"), legend.position='bottom')
    
    pblist[[i]]=ggplot(data=dat,aes_string(x='tsne1',y='tsne2'))+
      geom_point(aes(col=True_label),alpha=0.2)+
      # scale_color_manual(values = c("#999999", "#E69F00", "#56B4E9", "#009E73",
      #                               "#F0E442", "#0072B2", "#D55E00", "#CC79A7"))+
      scale_color_manual(values = c("gray70", "red"))+
      xlim(-60,60)+
      ylim(-60,60)+
      theme_bw() + theme(panel.border = element_blank(), panel.grid.major = element_blank(),
                         panel.grid.minor = element_blank(), 
                         axis.line = element_line(colour = "black"), legend.position='bottom')
  }
  
  p3=ggplot(data=dat,aes_string(x='tsne1',y='tsne2'))+
    geom_point(aes(col=slide),alpha=0.2)+
    xlim(-60,60)+
    ylim(-60,60)+ 
    theme_bw() + theme(panel.border = element_blank(), panel.grid.major = element_blank(),
                       panel.grid.minor = element_blank(), 
                       axis.line = element_line(colour = "black"), legend.position='none')
  
  p4=ggplot(data=subset(dat,True_label==TLB),aes_string(x='tsne1',y='tsne2'))+
    geom_point(aes(col=slide),alpha=0.2)+
    xlim(-60,60)+
    ylim(-60,60)+
    theme_bw() + theme(panel.border = element_blank(), panel.grid.major = element_blank(),
                       panel.grid.minor = element_blank(), 
                       axis.line = element_line(colour = "black"), legend.position='none')
  
  pdf(file=out_fig,
      width=14,height=7)
  
  for(i in 1:length(palist)){
    grid.arrange(palist[[i]],pblist[[i]],nrow=1)
  }
  grid.arrange(p3,p4,nrow=1)
  
  dev.off()
  
}

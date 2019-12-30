## reduce dimensionality of the acitvation layer of model
## visualize the manifold

# Old: I START AT 9, X START AT 12; ST start I at 11, X at 14
inlist=c('F4MSIst')

for(xx in inlist){
  load_file=paste('/Users/rh2740/documents/CPTAC-UCEC/Results/NL5/',xx,'/out/tSNE_P_N.csv',sep='')
  out_fig=paste('/Users/rh2740/documents/CPTAC-UCEC/Results/NL5/',xx,'/out/Figure.pdf',sep='')
  bins=50
  POS_score=c('MSI.H_score')
  MDP = 0.5 # 0.5 for binary; 1/length(POS_score))

  dat=read.table(load_file, header=T,sep=',')
  ## plot the manifold with probability
  library(ggplot2)
  library(gridExtra)
  palist <- list()
  for(i in 1:length(POS_score)){
    palist[[i]]=ggplot(data=dat,aes_string(x='tsne1',y='tsne2',col=POS_score[i]))+
      scale_color_gradient2(high='red',mid='gray',low='steelblue',midpoint=MDP)+
      geom_point(alpha=1, size=1)+ scale_shape(solid = TRUE)+
      #theme(legend.position='bottom')+
      xlim(-60,60)+
      ylim(-60,60)+
      theme_bw() + theme(panel.border = element_blank(), panel.grid.major = element_blank(),
                         panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))
  }
  
  pdf(file=out_fig,
      width=10,height=10,useDingbats=FALSE)
  
  grid.arrange(palist[[1]], nrow=1)
  
  dev.off()
}

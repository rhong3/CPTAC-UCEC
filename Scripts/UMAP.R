## reduce dimensionality of the acitvation layer of model
## visualize the manifold UMAP version

# args = commandArgs(trailingOnly=FALSE)
# input_file=args[1]
# output_file=args[2]
# out_fig=args[3]
# start=args[4]
# bins=args[5]
# POS_score=args[6]

input_file='/Users/rh2740/documents/CPTAC-UCEC/Results/NL4/X4TP53/out/For_tSNE.csv'
output_file='/Users/rh2740/documents/CPTAC-UCEC/Results/NL4/X4TP53/out/UMAP_P_N.csv'
out_fig='/Users/rh2740/documents/CPTAC-UCEC/Results/NL4/X4TP53/out/UMAP_P_N.pdf'
start=10
bins=50
POS_score='POS_score'

library(umap)
library(stats)
ori_dat = read.table(file=input_file,header=T,sep=',')
# P = ori_dat[which(ori_dat$Prediction==1),]
# N = ori_dat[which(ori_dat$Prediction==0),]
# N = ori_dat[sample(nrow(N), 20000), ]
# sp_ori_dat = rbind(P, N)
sp_ori_dat=ori_dat[sample(nrow(ori_dat), 20000), ]
# sp_ori_dat=ori_dat
X = as.matrix(sp_ori_dat[,start:dim(sp_ori_dat)[2]])
PCAX = prcomp(X)
newX = PCAX$x[,1:100]
res = umap(newX)
Y = res$layout
out_dat = cbind(sp_ori_dat[,1:(start-1)],Y)

dat = cbind(out_dat,x_bin=cut(out_dat[,start],bins),
            y_bin=cut(out_dat[,(start+1)],bins))

dat = cbind(dat, x_int = as.numeric(dat$x_bin),
            y_int = as.numeric(dat$y_bin))

colnames(dat)[start:(start+1)]=c('umap1','umap2')

dat$True_label=as.factor(dat$True_label)
dat$slide=as.factor(dat$slide)

write.table(dat, file=output_file, row.names = F, sep=',')

## plot the manifold with probability
library(ggplot2)
library(gridExtra)

p1=ggplot(data=dat,aes(x=umap1,y=umap2,col=POS_score))+
  scale_color_gradient2(high='darkorange',mid='white',low='steelblue',midpoint=0.5)+
  geom_point(alpha=0.2)+
  #theme(legend.position='bottom')+
  xlim(-15,15)+
  ylim(-15,15)

p2=ggplot(data=dat,aes(x=umap1,y=umap2))+
  geom_point(aes(col=True_label),alpha=0.2)+
  scale_color_manual(values = c('gray70','red'))+
  #theme(legend.position='bottom')+
  xlim(-15,15)+
  ylim(-15,15)

p3=ggplot(data=dat,aes(x=umap1,y=umap2))+
  geom_point(aes(col=slide),alpha=0.2)+
  theme(legend.position='none')+
  xlim(-15,15)+
  ylim(-15,15)

p4=ggplot(data=subset(dat,True_label==1),aes(x=umap1,y=umap2))+
  geom_point(aes(col=slide),alpha=0.2)+
  theme(legend.position='none')+
  xlim(-15,15)+
  ylim(-15,15)

pdf(file=out_fig,
    width=14,height=7)

grid.arrange(p1,p2,nrow=1)
grid.arrange(p3,p4,nrow=1)

dev.off()

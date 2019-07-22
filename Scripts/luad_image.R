## reduce dimensionality of the acitvation layer of model
## visualize the manifold

filename='tSNE_results.csv'
bins = 50

dat = read.table(filename, header = T, sep=',')

dat = cbind(dat,x_bin=cut(dat[,9],bins),
                y_bin=cut(dat[,10],bins))

dat = cbind(dat, x_int = as.numeric(dat$x_bin),
                 y_int = as.numeric(dat$y_bin))

colnames(dat)[9:10]=c('tsne1','tsne2')

dat$True_label=as.factor(dat$True_label)
dat$slide=as.factor(dat$slide)

## plot the manifold with probability
library(ggplot2)
library(gridExtra)

p1=ggplot(data=dat,aes(x=tsne1,y=tsne2,col=POS_score))+
  scale_color_gradient2(high='darkorange',mid='white',low='steelblue',midpoint=0.5)+
  geom_point(alpha=0.2)+
  #theme(legend.position='bottom')+
  xlim(-60,60)+
  ylim(-60,60)

p2=ggplot(data=dat,aes(x=tsne1,y=tsne2))+
  geom_point(aes(col=True_label),alpha=0.2)+
  scale_color_manual(values = c('gray70','red'))+
  #theme(legend.position='bottom')+
  xlim(-60,60)+
  ylim(-60,60)

p3=ggplot(data=dat,aes(x=tsne1,y=tsne2))+
  geom_point(aes(col=slide),alpha=0.2)+
  theme(legend.position='none')+
  xlim(-60,60)+
  ylim(-60,60)

p4=ggplot(data=subset(dat,True_label==1),aes(x=tsne1,y=tsne2))+
  geom_point(aes(col=slide),alpha=0.2)+
  theme(legend.position='none')+
  xlim(-60,60)+
  ylim(-60,60)

pdf(file='LUAD_test_manifold.pdf',
    width=14,height=7)

grid.arrange(p1,p2,nrow=1)
grid.arrange(p3,p4,nrow=1)

dev.off()

## random select representative images and output the file paths

select=data.frame()

for (i in 1:bins){
  for (j in 1:bins){
    sub = droplevels(subset(dat,x_int==i&y_int==j&True_label==0&POS_score<0.1))
    if(dim(sub)[1]>0){
      #print(dim(sub)[1])
      rownames(sub)=1:dim(sub)[1]
      ind=sample(1:dim(sub)[1],size=1)
      print(ind)
      impath=as.character(sub$path[ind])
      nl=data.frame(impath=impath,x_int=i,y_int=j)
      select=rbind(select,nl) 
    }
    rm(sub)
  }
}

select$impath=gsub('\\.\\.','/media/data01/Runyu/LUAD',select$impath)

write.table(select, file='selected_tiles_neg.csv',sep=',',quote=F,row.names=F)

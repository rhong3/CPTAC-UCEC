#!/usr/bin/env Rscript
## TO BE MODIFIED

args = commandArgs(trailingOnly=TRUE)
input_file=args[1]
output_file=args[2]

library(Rtsne)
dat = read.table(file=input_file,header=T,sep=',')
X = as.matrix(dat[,9:dim(dat)[2]])
res = Rtsne(X, initial_dims=100)
Y=res$Y
out_dat = cbind(dat[,1:8],Y)
write.table(out_dat, file=output_file, row.names = F, sep=',')
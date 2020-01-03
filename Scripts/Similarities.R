library(Hmisc)
library(Rfast)
library(dplyr)
library(pheatmap)
library(ComplexHeatmap)
# Correlation check
dict=read.csv('~/Documents/CPTAC-UCEC/dummy_His_MUT_joined.csv')
dict$subtype_Endometrioid = dict$subtype_Endometrioid-dict$subtype_0NA
dict$subtype_Endometrioid = gsub(-1, NA, dict$subtype_Endometrioid)
dict$subtype_MSI = dict$subtype_MSI-dict$subtype_0NA
dict$subtype_MSI = gsub(-1, NA, dict$subtype_MSI)
dict$subtype_POLE = dict$subtype_POLE-dict$subtype_0NA
dict$subtype_POLE = gsub(-1, NA, dict$subtype_POLE)
dict$subtype_Serous.like = dict$subtype_Serous.like-dict$subtype_0NA
dict$subtype_Serous.like = gsub(-1, NA, dict$subtype_Serous.like)
dict$MSIst_MSI.H = dict$MSIst_MSI.H-dict$MSIst_0NA
dict$MSIst_MSI.H = gsub(-1, NA, dict$MSIst_MSI.H)
dict$MSIst_MSS = dict$MSIst_MSS-dict$MSIst_0NA
dict$MSIst_MSS = gsub(-1, NA, dict$MSIst_MSS)
dict = subset(dict, select = -c(name, subtype_0NA, MSIst_0NA, histology_Mixed, MLH1, ERBB2, EGFR, PIK3R2, FAT4, ARID5B))

# Yule's Y Coefficient of colligation column-wise calculation 
YuleY = function(col1, col2){
  newdf=data.frame(cbind(col1, col2))
  colnames(newdf) = c('A', 'B')
  newdf=na.omit(newdf)
  mtx = matrix(0, nrow=2, ncol=2)
  mtx[1,1] = nrow(newdf[which(newdf$A == 0 & newdf$B == 0),])
  mtx[1,2] = nrow(newdf[which(newdf$A == 0 & newdf$B == 1),])
  mtx[2,1] = nrow(newdf[which(newdf$A == 1 & newdf$B == 0),])
  mtx[2,2] = nrow(newdf[which(newdf$A == 1 & newdf$B == 1),])
  return(list('yu' = yule(mtx), 'count' = sum(mtx)))
}

OUTPUT = setNames(data.frame((matrix(ncol = ncol(dict), nrow = ncol(dict)))), colnames(dict))
INTER = setNames(data.frame((matrix(ncol = ncol(dict), nrow = ncol(dict)))), colnames(dict))

row.names(OUTPUT) = colnames(OUTPUT)
row.names(INTER) = colnames(INTER)
for (i in 1:ncol(dict)){
  for (j in 1:ncol(dict)){
    rtt = YuleY(dict[,i], dict[,j])
    OUTPUT[colnames(dict)[i], colnames(dict)[j]] = rtt$yu
    INTER[colnames(dict)[i], colnames(dict)[j]] = rtt$count
  }
}

OUTPUT[is.na(OUTPUT)] = 1


OUTPUT_hm = OUTPUT
OUTPUT = round(OUTPUT, digits = 2)
write.csv(OUTPUT, '~/Documents/CPTAC-UCEC/YuleY_similarities.csv', row.names=TRUE)

colnames(OUTPUT_hm) = gsub('histology_','', colnames(OUTPUT_hm))
rownames(OUTPUT_hm) = gsub('histology_','', rownames(OUTPUT_hm))
# colnames(OUTPUT_hm) = gsub('subtype','ST', colnames(OUTPUT_hm))
# rownames(OUTPUT_hm) = gsub('subtype','ST', rownames(OUTPUT_hm))
colnames(OUTPUT_hm) = gsub('MSIst_','', colnames(OUTPUT_hm))
rownames(OUTPUT_hm) = gsub('MSIst_','', rownames(OUTPUT_hm))
colnames(OUTPUT_hm) = gsub('MSI.H','MSI-H', colnames(OUTPUT_hm))
rownames(OUTPUT_hm) = gsub('MSI.H','MSI-H', rownames(OUTPUT_hm))
colnames(OUTPUT_hm) = gsub('MSI.H','MSI-H', colnames(OUTPUT_hm))
rownames(OUTPUT_hm) = gsub('MSI.H','MSI-H', rownames(OUTPUT_hm))
# colnames(OUTPUT_hm) = gsub('Endometrioid','Endo', colnames(OUTPUT_hm))
# rownames(OUTPUT_hm) = gsub('Endometrioid','Endo', rownames(OUTPUT_hm))
colnames(OUTPUT_hm) = gsub('subtype_Serous.like','CNV-H', colnames(OUTPUT_hm))
rownames(OUTPUT_hm) = gsub('subtype_Serous.like','CNV-H', rownames(OUTPUT_hm))
colnames(OUTPUT_hm) = gsub('subtype_Endometrioid','CNV-L', colnames(OUTPUT_hm))
rownames(OUTPUT_hm) = gsub('subtype_Endometrioid','CNV-L', rownames(OUTPUT_hm))
colnames(OUTPUT_hm) = gsub('subtype_MSI','MSI', colnames(OUTPUT_hm))
rownames(OUTPUT_hm) = gsub('subtype_MSI','MSI', rownames(OUTPUT_hm))
colnames(OUTPUT_hm) = gsub('subtype_POLE','POLE', colnames(OUTPUT_hm))
rownames(OUTPUT_hm) = gsub('subtype_POLE','POLE', rownames(OUTPUT_hm))

OUTPUT_hm_full = OUTPUT_hm

OUTPUT_hm_inter = OUTPUT_hm[19:24, 1:18]

OUTPUT_hm_gene = OUTPUT_hm[1:18, 1:18]


out_fig='~/Documents/CPTAC-UCEC/Inter_YuleY_similarities.pdf'
pdf(file=out_fig,
    width=12,height=4.5)
myBreaks <- c(seq(min(OUTPUT_hm), 0, length.out=ceiling(100/2) + 1), 
              seq(max(OUTPUT_hm)/100, max(OUTPUT_hm), length.out=floor(100/2)))
pheatmap(OUTPUT_hm_inter, cluster_cols = FALSE, cluster_rows = FALSE, main = "YuleY Colligation", breaks = myBreaks)
dev.off()

nm = rownames(OUTPUT_hm_inter)
col_fun = circlize::colorRamp2(c(-1, 0, 1), c( "#4575B4", "#ffffff", "#D73027"))
# `col = col_fun` here is used to generate the legend
dev.off()

out_fig='~/Documents/CPTAC-UCEC/Inter_YuleY_similarities_COMPLEX.pdf'
pdf(file=out_fig,
    width=20,height=15)
Heatmap(OUTPUT_hm_inter, column_title = "YuleY Colligation", name = "colligation", col = col_fun, rect_gp = gpar(type = "none"), 
        cell_fun = function(j, i, x, y, width, height, fill) {
          grid.rect(x = x, y = y, width = width, height = height, 
                    gp = gpar(col = "grey", fill = NA))
          if(i == j) {
            grid.text(nm[i], x = x, y = y, gp = gpar(fontsize = 10))
          } else if(i > j) {
            grid.circle(x = x, y = y, r = abs(INTER[i, j])/max(INTER)/2 * min(unit.c(width, height)), 
                        gp = gpar(fill = col_fun(OUTPUT_hm_inter[i, j]), col = NA))
          } else {
            grid.text(sprintf("%.2f", OUTPUT_hm_inter[i, j]), x, y, gp = gpar(fontsize = 10))
          }
        }, cluster_rows = FALSE, cluster_columns = FALSE,
        show_row_names = FALSE, show_column_names = FALSE)
dev.off()



out_fig='~/Documents/CPTAC-UCEC/Gene_YuleY_similarities.pdf'
pdf(file=out_fig,
    width=8.5,height=7)
myBreaks <- c(seq(min(OUTPUT_hm), 0, length.out=ceiling(100/2) + 1), 
              seq(max(OUTPUT_hm)/100, max(OUTPUT_hm), length.out=floor(100/2)))
pheatmap(OUTPUT_hm_gene, cluster_cols = FALSE, cluster_rows = FALSE, main = "YuleY Colligation", breaks = myBreaks)
dev.off()

nm = rownames(OUTPUT_hm_gene)
col_fun = circlize::colorRamp2(c(-1, 0, 1), c( "#4575B4", "#ffffff", "#D73027"))
# `col = col_fun` here is used to generate the legend

out_fig='~/Documents/CPTAC-UCEC/Gene_YuleY_similarities_COMPLEX.pdf'
pdf(file=out_fig,
    width=20,height=15)
Heatmap(OUTPUT_hm_gene, column_title = "YuleY Colligation", name = "colligation", col = col_fun, rect_gp = gpar(type = "none"), 
        cell_fun = function(j, i, x, y, width, height, fill) {
          grid.rect(x = x, y = y, width = width, height = height, 
                    gp = gpar(col = "grey", fill = NA))
          if(i == j) {
            grid.text(nm[i], x = x, y = y, gp = gpar(fontsize = 10))
          } else if(i > j) {
            grid.circle(x = x, y = y, r = abs(INTER[i, j])/max(INTER)/2 * min(unit.c(width, height)), 
                        gp = gpar(fill = col_fun(OUTPUT_hm_gene[i, j]), col = NA))
          } else {
            grid.text(sprintf("%.2f", OUTPUT_hm_gene[i, j]), x, y, gp = gpar(fontsize = 10))
          }
        }, cluster_rows = FALSE, cluster_columns = FALSE,
        show_row_names = FALSE, show_column_names = FALSE)
dev.off()


out_fig='~/Documents/CPTAC-UCEC/YuleY_similarities.pdf'
pdf(file=out_fig,
    width=8.5,height=7)
myBreaks <- c(seq(min(OUTPUT_hm), 0, length.out=ceiling(100/2) + 1), 
              seq(max(OUTPUT_hm)/100, max(OUTPUT_hm), length.out=floor(100/2)))
pheatmap(OUTPUT_hm_full, cluster_cols = FALSE, cluster_rows = FALSE, main = "YuleY Colligation", breaks = myBreaks)
dev.off()

nm = rownames(OUTPUT_hm_full)
col_fun = circlize::colorRamp2(c(-1, 0, 1), c( "#4575B4", "#ffffff", "#D73027"))
# `col = col_fun` here is used to generate the legend

out_fig='~/Documents/CPTAC-UCEC/YuleY_similarities_COMPLEX.pdf'
pdf(file=out_fig,
    width=20,height=15)
Heatmap(OUTPUT_hm_full, column_title = "YuleY Colligation", name = "colligation", col = col_fun, rect_gp = gpar(type = "none"), 
        cell_fun = function(j, i, x, y, width, height, fill) {
          grid.rect(x = x, y = y, width = width, height = height, 
                    gp = gpar(col = "grey", fill = NA))
          if(i == j) {
            grid.text(nm[i], x = x, y = y, gp = gpar(fontsize = 10))
          } else if(i > j) {
            grid.circle(x = x, y = y, r = abs(INTER[i, j])/max(INTER)/2 * min(unit.c(width, height)), 
                        gp = gpar(fill = col_fun(OUTPUT_hm_full[i, j]), col = NA))
          } else {
            grid.text(sprintf("%.2f", OUTPUT_hm_full[i, j]), x, y, gp = gpar(fontsize = 10))
          }
        }, cluster_rows = FALSE, cluster_columns = FALSE,
        show_row_names = FALSE, show_column_names = FALSE)
dev.off()


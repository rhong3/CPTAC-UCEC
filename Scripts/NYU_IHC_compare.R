# NYU IHC compare
library(dplyr)
library(ggplot2)
library(readr)

# P53 features
filter = read.csv("~/documents/CPTAC-UCEC/Results/Statistics_NYU_IHC_filter.csv")
all = read.csv("~/documents/CPTAC-UCEC/Results/Statistics_NYU_IHC.csv")
NYU = read.csv("~/documents/CPTAC-UCEC/Results/Statistics_NYU.csv")

filter$Feature = gsub("_P53", "", filter$Feature)
all$Feature = gsub("_P53", "", all$Feature)

NYU = NYU %>%
  select(c(1:6,28:30)) %>%
  mutate(group="RAW")

filter = filter %>%
  select(c(1:6,28:30)) %>%
  mutate(group="IHC")

all = all %>%
  select(c(1:6,28:30)) %>%
  mutate(group="Mix")

concat = rbind(NYU, filter)
concat = rbind(concat, all)
concat = concat[concat$Feature %in% c("CNVH", "TP53"),]

concat$Architecture = gsub("X1", "P2", concat$Architecture)
concat$Architecture = gsub("X2", "P1", concat$Architecture)
concat$Architecture = gsub("X3", "P4", concat$Architecture)
concat$Architecture = gsub("X4", "P3", concat$Architecture)
concat$Architecture = gsub("F1", "PC2", concat$Architecture)
concat$Architecture = gsub("F2", "PC1", concat$Architecture)
concat$Architecture = gsub("F3", "PC4", concat$Architecture)
concat$Architecture = gsub("F4", "PC3", concat$Architecture)
concat$Tiles = gsub("NL5", "TCGA+CPTAC", concat$Tiles)
concat$Tiles = gsub("NL6", "TCGA", concat$Tiles)


concat.CNVH = concat[concat$Feature == "CNVH",]
concat.TP53 = concat[concat$Feature == "TP53",]

pdf(file="~/documents/CPTAC-UCEC/Results/IHC_CNVH_test_comparison.pdf",
    width=14,height=7)

ggplot(concat.CNVH, aes(Architecture, Patient_ROC, fill=group)) +
geom_bar(position="dodge", stat="identity") + 
geom_errorbar(aes(ymin=Patient_ROC.95.CI_lower, ymax=Patient_ROC.95.CI_upper), width=.2,
                position=position_dodge(.9)) + scale_fill_brewer(palette="Dark2")  +  
  ggtitle("CNVH Per-Patient Comparison") + facet_wrap(~Tiles) + theme_classic()

ggplot(concat.CNVH, aes(Architecture, Tile_ROC, fill=group)) +
  geom_bar(position="dodge", stat="identity") + 
  geom_errorbar(aes(ymin=Tile_ROC.95.CI_lower, ymax=Tile_ROC.95.CI_upper), width=.2,
                position=position_dodge(.9)) + scale_fill_brewer(palette="Dark2")  +  
  ggtitle("CNVH Per-Tile Comparison") + facet_wrap(~Tiles) + theme_classic()

dev.off()

pdf(file="~/documents/CPTAC-UCEC/Results/IHC_TP53_test_comparison.pdf",
    width=14,height=7)

ggplot(concat.TP53, aes(Architecture, Patient_ROC, fill=group)) +
  geom_bar(position="dodge", stat="identity") + 
  geom_errorbar(aes(ymin=Patient_ROC.95.CI_lower, ymax=Patient_ROC.95.CI_upper), width=.2,
                position=position_dodge(.9)) + scale_fill_brewer(palette="Dark2")  +  
  ggtitle("TP53 Per-Patient Comparison") + facet_wrap(~Tiles) + theme_classic()

ggplot(concat.TP53, aes(Architecture, Tile_ROC, fill=group)) +
  geom_bar(position="dodge", stat="identity") + 
  geom_errorbar(aes(ymin=Tile_ROC.95.CI_lower, ymax=Tile_ROC.95.CI_upper), width=.2,
                position=position_dodge(.9)) + scale_fill_brewer(palette="Dark2")  +  
  ggtitle("TP53 Per-Tile Comparison") + facet_wrap(~Tiles) + theme_classic()

dev.off()


# MSI features
library(dplyr)
library(ggplot2)
library(readr)

for (fff in c('_PMS2', '_MLH1', '_MSH2', '_MSH6', '_PMS2-MLH1', '_MSH2-MSH6', 
              '_PMS2-MLH1-MSH2-MSH6')){
  filter = read.csv("~/documents/CPTAC-UCEC/Results/Statistics_NYU_IHC_filter.csv")
  all = read.csv("~/documents/CPTAC-UCEC/Results/Statistics_NYU_IHC.csv")
  NYU = read.csv("~/documents/CPTAC-UCEC/Results/Statistics_NYU.csv")
  
  filter$Feature = gsub(fff, "", filter$Feature)
  all$Feature = gsub(fff, "", all$Feature)
  
  NYU = NYU %>%
    select(c(1:6,28:30)) %>%
    mutate(group="RAW")
  
  filter = filter %>%
    select(c(1:6,28:30)) %>%
    mutate(group="IHC")
  
  all = all %>%
    select(c(1:6,28:30)) %>%
    mutate(group="Mix")
  
  concat = rbind(NYU, filter)
  concat = rbind(concat, all)
  
  concat$Architecture = gsub("X1", "P2", concat$Architecture)
  concat$Architecture = gsub("X2", "P1", concat$Architecture)
  concat$Architecture = gsub("X3", "P4", concat$Architecture)
  concat$Architecture = gsub("X4", "P3", concat$Architecture)
  concat$Architecture = gsub("F1", "PC2", concat$Architecture)
  concat$Architecture = gsub("F2", "PC1", concat$Architecture)
  concat$Architecture = gsub("F3", "PC4", concat$Architecture)
  concat$Architecture = gsub("F4", "PC3", concat$Architecture)
  concat$Tiles = gsub("NL5", "TCGA+CPTAC", concat$Tiles)
  concat$Tiles = gsub("NL6", "TCGA", concat$Tiles)
  
  concat = concat[concat$Feature == "MSI",]
  
  pdf(file=paste("~/documents/CPTAC-UCEC/Results/IHC_MSI", fff,"_test_comparison.pdf", sep=''),
      width=14,height=7)
  
  print(ggplot(concat, aes(Architecture, Patient_ROC, fill=group)) +
    geom_bar(position="dodge", stat="identity") + 
    geom_errorbar(aes(ymin=Patient_ROC.95.CI_lower, ymax=Patient_ROC.95.CI_upper), width=.2,
                  position=position_dodge(.9)) + scale_fill_brewer(palette="Dark2")  +  
    ggtitle(paste('MSI', fff, " Per-Patient Comparison", sep='')) + facet_wrap(~Tiles) + theme_classic())
  
  print(ggplot(concat, aes(Architecture, Tile_ROC, fill=group)) +
    geom_bar(position="dodge", stat="identity") + 
    geom_errorbar(aes(ymin=Tile_ROC.95.CI_lower, ymax=Tile_ROC.95.CI_upper), width=.2,
                  position=position_dodge(.9)) + scale_fill_brewer(palette="Dark2")  +  
    ggtitle(paste('MSI', fff, " Per-Tile Comparison", sep='')) + facet_wrap(~Tiles) + theme_classic())
  
  dev.off() 
  
}


# MSI features side by side
library(dplyr)
library(ggplot2)
library(readr)
library(gridExtra)

all = read.csv("~/documents/CPTAC-UCEC/Results/Statistics_NYU_IHC.csv")
filter = read.csv("~/documents/CPTAC-UCEC/Results/Statistics_NYU_IHC_filter.csv")

filter = filter %>%
  select(c(1:6,28:30))

all = all %>%
  select(c(1:6,28:30))

# all
all$Architecture = gsub("X1", "P2", all$Architecture)
all$Architecture = gsub("X2", "P1", all$Architecture)
all$Architecture = gsub("X3", "P4", all$Architecture)
all$Architecture = gsub("X4", "P3", all$Architecture)
all$Architecture = gsub("F1", "PC2", all$Architecture)
all$Architecture = gsub("F2", "PC1", all$Architecture)
all$Architecture = gsub("F3", "PC4", all$Architecture)
all$Architecture = gsub("F4", "PC3", all$Architecture)
all$Tiles = gsub("NL5", "TCGA+CPTAC", all$Tiles)
all$Tiles = gsub("NL6", "TCGA", all$Tiles)

all = all[grepl("MSI", all$Feature),]



pa = ggplot(all, aes(Architecture, Patient_ROC, fill=Feature)) +
  geom_bar(position="dodge", stat="identity") + 
  geom_errorbar(aes(ymin=Patient_ROC.95.CI_lower, ymax=Patient_ROC.95.CI_upper), width=.2,
                position=position_dodge(.9)) + scale_fill_brewer(palette="Dark2")  +  
  ggtitle("MSI Markers Per-Patient Comparison") + facet_wrap(~Tiles) + theme_classic()

pb = ggplot(all, aes(Architecture, Tile_ROC, fill=Feature)) +
  geom_bar(position="dodge", stat="identity") + 
  geom_errorbar(aes(ymin=Tile_ROC.95.CI_lower, ymax=Tile_ROC.95.CI_upper), width=.2,
                position=position_dodge(.9)) + scale_fill_brewer(palette="Dark2")  +  
  ggtitle("MSI Markers Per-Tile Comparison") + facet_wrap(~Tiles) + theme_classic()

pdf(file=paste("~/documents/CPTAC-UCEC/Results/IHC_MSI_markers_test_comparison_all.pdf", sep=''),
    width=30,height=15)

grid.arrange(pa,pb,ncol=1)

dev.off()

# filtered
filter$Architecture = gsub("X1", "P2", filter$Architecture)
filter$Architecture = gsub("X2", "P1", filter$Architecture)
filter$Architecture = gsub("X3", "P4", filter$Architecture)
filter$Architecture = gsub("X4", "P3", filter$Architecture)
filter$Architecture = gsub("F1", "PC2", filter$Architecture)
filter$Architecture = gsub("F2", "PC1", filter$Architecture)
filter$Architecture = gsub("F3", "PC4", filter$Architecture)
filter$Architecture = gsub("F4", "PC3", filter$Architecture)
filter$Tiles = gsub("NL5", "TCGA+CPTAC", filter$Tiles)
filter$Tiles = gsub("NL6", "TCGA", filter$Tiles)

filter = filter[grepl("MSI", filter$Feature),]

pa = ggplot(filter, aes(Architecture, Patient_ROC, fill=Feature)) +
  geom_bar(position="dodge", stat="identity") + 
  geom_errorbar(aes(ymin=Patient_ROC.95.CI_lower, ymax=Patient_ROC.95.CI_upper), width=.2,
                position=position_dodge(.9)) + scale_fill_brewer(palette="Dark2")  +  
  ggtitle("MSI Markers Per-Patient Comparison") + facet_wrap(~Tiles) + theme_classic()

pb = ggplot(filter, aes(Architecture, Tile_ROC, fill=Feature)) +
  geom_bar(position="dodge", stat="identity") + 
  geom_errorbar(aes(ymin=Tile_ROC.95.CI_lower, ymax=Tile_ROC.95.CI_upper), width=.2,
                position=position_dodge(.9)) + scale_fill_brewer(palette="Dark2")  +  
  ggtitle("MSI Markers Per-Tile Comparison") + facet_wrap(~Tiles) + theme_classic()

pdf(file=paste("~/documents/CPTAC-UCEC/Results/IHC_MSI_markers_test_comparison_filter.pdf", sep=''),
    width=30,height=15)

grid.arrange(pa,pb,ncol=1)

dev.off()



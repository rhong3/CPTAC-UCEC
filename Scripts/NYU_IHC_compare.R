# NYU IHC compare
library(dplyr)
library(ggplot2)
library(readr)

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

pdf(file="~/documents/CPTAC-UCEC/Results/IHC_test_comparison.pdf",
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



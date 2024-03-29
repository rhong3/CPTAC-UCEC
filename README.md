### Endometrial Cancer Molecular Features prediction and Visualization Using Deep Learning
https://doi.org/10.1016/j.xcrm.2021.100400
# Features included 
 - 18 mutations (ARID1A, ATM, BRCA2, CTCF, CTNNB1, FAT1, FBXW7, FGFR2, JAK1, KRAS, MTOR, 
 PIK3CA, PIK3R1, PPP2R1A, PTEN, RPL22, TP53, ZFHX3)
 - 4 molecular subtypes (CNV.H, CNV.L, MSI, POLE)
 - Histological subtypes (Endometrioid, Serous)
   
## Architecture included
 - Inception (V1, V2, V3, V4)
 - inception-ResNet (V1, V2)
 - Panoptes(X & F) (V1, V2, V3, V4)
 
## Catalogue of codes including all statistical analyses codes
 - Accessory.py: Accessory functions for Inception Models, including AUROC/AUPRC plotting, CAM, etc.
 - Accessory2.py: Accessory functions for X Models, including AUROC/AUPRC plotting, CAM, etc.
 - annotation_plot.R: Build data summary heatmap.
 - Cutter.py: Bulk cutting svs images and normalization.
 - Cutter_NYU.py: Bulk cutting svs images and normalization for NYU samples.
 - cnn4.py: Tensorflow driving code for single resolution model training/validation/testing.
 - cnn5.py: Tensorflow driving code for multi resolution model training/validation/testing.
 - data_input2.py: Reading TFrecords file for single resolution tiles.
 - data_input3.py: Reading TFrecords file for multi resolution tiles.
 - data_input_fusion.py: Reading TFrecords file for multi resolution tiles with BMI and age.
 - Fusion_prep.py: Label preparation with BMI and age integrated.
 - HE_mosaic.py: Make tSNE mosaic plots (single resolution).
 - HE_mosaic2.py: Make tSNE mosaic plots (multi resolution).
 - Img_summary.R: Summary of images in the cohort (dimension, # images per patient, counts).
 - InceptionV1.py: InceptionV1 architecture.
 - InceptionV2.py: InceptionV2 architecture.
 - InceptionV3.py: InceptionV3 architecture.
 - InceptionV4.py: InceptionV4 architecture.
 - InceptionV5.py: InceptionResnetV1 architecture.
 - InceptionV6.py: InceptionResnetV2 architecture.
 - Label_prep2.py: Label preparation.
 - mainm3.py: Main method for single resolution model training/validation/testing.
 - mainm4.py: Main method for multi resolution model training/validation/testing. 
 - make_table.R: Summarize all the prediction tasks results in a table and a heatmap.
 - MW_test.R: Tile-level Wilcoxon tests and plotting.
 - Model_stat_test.R: Patient-level Wilcoxon tests, t-tests, AUROC tests, and plotting.
 - multi_stat_test.R: compare multi-resolution and single resolution models.
 - NYU_data_prep.py: NYU data preparation.
 - NYU_loaders.py: Loading NYU data.
 - NYU_test.py: Run testing on NYU dataset.
 - POLE_pred.R: Multi-model systems to predict POLE subtype.  
 - RGB_profiler.py: Get RGB summary of tiles in cohort. 
 - RealtestV4.py: Deployment code for trained models.
 - Realtest_for_figure.py: Deployment code for best performing trained models.
 - ROC_figure.R: Example ROC plot for figures.
 - Sample_prep.py: Sample preparation code for single resolution models, including sampling.
 - Sample_prep2.py: Sample preparation code for multi resolution models, including sampling.
 - scatter_logits.R: Plot prediction logits for figure 6.
 - Similarities.R: YuleY similarity calculations between features.
 - Slicer.py: Multi-thread cutting of images.
 - Slicer_NYU.py: Multi-thread cutting of NYU images.   
 - Slide_Size_Count.py: Image dimension summary and number of tiles per image counts.
 - Statistics_MSI.R: Statistical metrics for MSI predictions.
 - Statistics_histology.R: Statistical metrics for histological subtype predictions.
 - Statistics_mutations.R: Statistical metrics for mutation predictions.
 - Statistics_special.R: Statistical metrics for other types of predictions.
 - Statistics_subtypes.R: Statistical metrics for molecular subtype predictions.
 - Statistics_NYU.R: Statistical metrics for NYU samples.
 - Summary.R: Count number of patients for each task in the cohort.
 - SummaryTable.R: Summary table for the paper.
 - tSNE.R: tSNE dimensional reduction for activation maps.
 - tSNE_for_figure.R: High quality tSNE dimensional reduction for activation maps.   
 - UMAP.R: UMAP dimensional reduction for activation maps.
 - X1.py: Panoptes2 architecture.
 - X2.py: Panoptes1 architecture.
 - X3.py: Panoptes4 architecture.
 - X4.py: Panoptes3 architecture.
 - Legacy: Deprecated codes.
 
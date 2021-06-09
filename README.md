# [Some name] a class to compare performance of AL and non-AL models

Example of command to run the pipeline:

```$  python main.py -s_n 'N_SF_TTS'```

I used [t-test for means of two independent samples](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind_from_stats.html), since our AL and non-AL models are trained on different data 
(AL training set is a subset of non-AL training set). 

# UML
![UML](Pipeline_UML.jpg?raw=true "Title")


# Research Summary

**hypothesis** *Training data sampling can significantly improve the performance of SCAM classification models*

**measure of success** We define a data sampling strategy to improve the performance if a machine learning model that includes the sampling strategy has a significantly better performance compared to the same machine learning model but without the sampling strategy. To evaluate such pairs of models, we will
- keep all other model parameters and pre-processing steps consistent (e.g. dataset, train-test split, parameter optimization strategy, used descriptors, feature selection strategy) 
- performance metric: ROC AUC, MCC, F1
- test set: 60-30% training-test split of original data, while ensuring consistent imbalance in test set (stratified) and using scaffold-based group assignment.
- we will repeat training-test split 10-times and use bonferroni corrected t-test p values to ensure differences are significant.

To ensure that our hypothesis is generalizable and not limited to a single use case, we will explore these different scenarios
- sampling strategies
  - ADASYN
  - SMOTE
  - CondensedNearestNeighbor
  - ActiveLearning
- dataset
   - small Shoichet dataset from Excel sheet
   - larger Shoichet dataset Excel + large set of positive data from AggAdvisor
   - dataset from Tropsha SCAMDetective based PubChem
     - bLactamase https://pubs-acs-org.proxy.lib.duke.edu/doi/suppl/10.1021/acs.jcim.0c00415/suppl_file/ci0c00415_si_002.zip
     - Cruzain https://pubs-acs-org.proxy.lib.duke.edu/doi/suppl/10.1021/acs.jcim.0c00415/suppl_file/ci0c00415_si_003.zip
- descriptor
  - ECFP (Morgan)
  - RDKit Fingerprint
- feature processing
  - none
  - feature scaling
- models
  - TF MLP
  - DeepSCAMs


# Results
Results could be found [here](Description/Update.pdf)


# Folder names

| Sampling           | Dataset  | Split |
|:-------------:| -----:|-----:|
| N (No sampling) | SF (SCAMS_filtered.csv) | TTS (train_test_split) | 
| SMOTE    |   SP1 (SCAMS_balanced_with_positive.csv) | B (split_with_butina) |
| ADASYN  |    SP2 (SCAMS_added_positives_653_1043.csv) | SS (split_with_scaffold_splitter) |
| CondensedNearestNeighbour (CNN)     |    __ | __ |
| InstanceHardnessThreshold (IHT)    |    __ | __ |

For example, N_SF_SS stands for run with with no sampling on SCAMS_filtered.csv and scaffold_splitter


# Other approaches
## SCAMs detective (SD)
There are two models presented with SD (**cruzain** and **beta-lactamase**). I made a mistake in the previous email and wrote that there were 4 models, as pbz2-files were not models.


### Results

| Model name        | AUC lower estimate | AUC  | AUC upper estimate | Accuracy |  F1 |  MCC
| ------------- |:-------------:| -----:|-----:| -----:| -----:| -----:| 
| cruzain |0.754| 0.851|0.948| 0.753| 0.652| 0.573| 
| beta-lactamase |0.556| 0.687 |0.819 | 0.569| 0.263| 0.165| 

As we can see from the results, the cruzain model performs better. I decided to make sure that the training sets do not contain compounds from the DeepSCAMs validation set (DLS) or very similar molecules. 
## DeepSCAMs (DC)

### Results
I trained in parallel  DC and our models, and visualized the results.


##### Radar-chart with mean values
![AL_non_AL_Deep_SCAMs_performance](Results/DeepSCAMs/AL_non_AL_Deep_SCAMs_performance.svg?raw=true "Title")




# DeepSCAMs validation set ∩ with the training datasets
- I [checked](/SCAMs/intersection_DeepSCAMs_ds_cruzian.py) the similarity of compounds in the DeepSCAMs validation set (DLS) with the cruzain dataset (used morgan fingerprints). Here is [the table](/Similarity/DLS_cruzian.csv). 17/65 compounds from the DLS dataset have one or more compounds in the cruzain dataset with Tanimoto coefficient (TC) > 0.7.
- Checked the similarity of compounds in the DLS dataset with the beta-lactamase dataset, found 18/65 highly similar compounds. [The resulting table is here](/Similarity/DLS_beta_lactamase.csv). Albeit, the performance of the model trained on the beta-lactamase dataset was lower.
- Checked the similarity of compounds in the DLS dataset with the SF dataset, found 2/65 compound with the same TC threshold. [The resulting table is here](/Similarity/DLS_SCAMS_filtered.csv).
- Checked SCAMS_added_positives_653_1043.csv vs. DLS. Found 4/64 with the same TC threshold.  [The resulting table is here](/Similarity/DLS_added_positives_653_1043.csv)
- Checked SCAMS_balanced_with_positive.csv vs. DLS. Found 2/64 with the same TC threshold.  [The resulting table is here](/Similarity/DLS_SCAMS_balanced_positive.csv)


# SCAMs

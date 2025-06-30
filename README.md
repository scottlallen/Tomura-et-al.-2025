# Ensemble

The shared code is used for "Improvements in Prediction Performance of Ensemble Approaches for Genomic Prediction in Crop Breeding" (https://doi.org/10.1093/g3journal/jkaf048).

Workflow:
1) Your target data should be preprocessed first. Imputation and pruning were applied to this experiment.
2) Install required libraries and packages (check the "environment" folder for the details).
3) Each individual genomic prediction model should return their genomic prediction results. These are used as input for an ensemble.

# glaucoma-incidence

### identifying incidence cases
`incident_glaucoma.ipynb`: identifies incident cases and sources, perform train/test split 

### identifying subcohorts and imputation
`getting_age_sex_matched_and_IOP_removed_subcohorts_and_imputation.ipynb`: matched and IOP removed subcohorts and their imputation \
`imputation_for_IOP_subcohort.ipynb`: IOP subcohort and its imputation \
`models/RNFL_hyperparameter_tuning.ipynb`: identifying RNFL subcohort and its imputation in first sections 

### feature sets
`feature_sets.py`: feature sets for IOP, IOP removed, and matched subcohorts \
`feature_sets_rnfl.py`: feature sets for RNFL subcohort 



## models

### hyperparameter tuning
`IOP_hyperparameter_tuning.ipynb`: hyperparameter tuning for IOP subcohort \
`IOPremoved_hyperparameter_tuning.ipynb`: hyperparameter tuning for IOP removed subcohort \
`RNFL_hyperparameter_tuning.ipynb`: hyperparameter tuning for RNFL subcohort 

### initial model evaluation
`model_evaluation_3year_tte.ipynb`: initial model evaluation for all 3 year subcohorts \
`model_evaluation_5year_tte.ipynb`: initial model evaluation for all 5 year subcohorts \
`model_evaluation_10year_tte.ipynb`: initial model evaluation for all 10 year subcohorts 

### final model evaluation
`minimal_feature_models.ipynb`: final model evaluation for all subcohorts 

### helper functions
`model_util.py`: hyperparameter tuning scoring metrics \
`optuna_util.py`: applying hyperparameter tuning ranges

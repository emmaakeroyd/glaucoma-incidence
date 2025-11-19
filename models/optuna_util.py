import numpy as np
import sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.kernel_approximation import RBFSampler, Nystroem
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.linear_model import SGDClassifier
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.linear_model import LogisticRegression
from imblearn.combine import SMOTEENN
import xgboost as xgb
import lightgbm as lgb
import importlib
import os
import math
import random
import plotly

import torch
import torch.nn as nn
from torch.nn import ReLU
import torch.multiprocessing as mp


import joblib
from joblib import dump, load

import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
from optuna.samplers import TPESampler




from joblib import Parallel, delayed

### for notebook
import model_util
from model_util import get_scoring_metrics


#from sklearnex import patch_sklearn
#patch_sklearn()



### Run multiple studies for different model feature sets
def run_optuna_studies(X, y, feature_dict, n_trials, objective_class, save_dir, **kwargs):

    study_dict = {}
    
    for model_name, features in feature_dict.items():
        #missing = [f for f in features if f not in X.columns]
        #if missing:
            #print(f"Warning: {len(missing)} features missing for {model_name}")
        print(f'\nRunning optuna study for {model_name}\n')

        model_dir = save_dir
        if save_dir != None:
            model_dir = f'{save_dir}/{model_name}'
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
        
        model_X = X[features]
        study = optuna.create_study(direction='maximize', sampler=TPESampler())
        objective = objective_class(model_X, y, **kwargs)
        study.optimize(objective, n_trials=n_trials)

        study_dict[model_name] = study

        study_df = study.trials_dataframe()
        #display(study_df) #DISABLED FOR TMUX TO RUN JUST PY

        hyperparam_importance_fig = plot_param_importances(study)
        optimization_history_fig = plot_optimization_history(study)
        
        if model_dir != None:
            dump(study, f'{model_dir}/optuna_study_{model_name}.pkl')

            study_df.to_csv(f'{model_dir}/optuna_study_{model_name}.csv', index=False)
            #hyperparam_importance_fig.write_image(f'{model_dir}/hyperparam_importance_{model_name}.png')
            #optimization_history_fig.write_image(f'{model_dir}/optimization_history_{model_name}.png')

        #hyperparam_importance_fig.show()
        #optimization_history_fig.show()

    return study_dict


# Base class
############################################### 

def fit(est, X, y):
    est.fit(X, y)

class OptunaObjective:
    def __init__(self, X, y, estimator_class, constant_param_dict, n_cv_folds=5, scoring_metric='roc_auc'):
        self.X = X
        self.y = y
        self.estimator_class = estimator_class
        self.constant_param_dict = constant_param_dict
        self.n_cv_folds = n_cv_folds
        self.scoring_metric = scoring_metric
        self.kf = StratifiedKFold(n_splits=n_cv_folds, shuffle=True, random_state=2024)

    def suggest_trial_params(self, trial):
        pass
    
    def __call__(self, trial):
        params = self.suggest_trial_params(trial)
        params.update(self.constant_param_dict)

        if trial.number == 0:
            print(trial.distributions)

        print(f'\nStarting trial {trial.number}')
        print(params)

        trial.set_user_attr('all_params', params)

        cv_eval_metrics = {
            'roc_auc': np.array([]),
            'optimal_threshold': np.array([]),
            'sensitivity_recall': np.array([]),
            'specificity': np.array([]),
            'ppv_precision': np.array([]),
            'npv': np.array([]),
            'f1_score': np.array([]),
            'pr_auc': np.array([]),
        }

        def run_cv(cv_i, indices):
            train_indices, validation_indices = indices
            X_cv_train = self.X.iloc[train_indices]
            y_cv_train  = self.y.iloc[train_indices]
            X_cv_val = self.X.iloc[validation_indices]
            y_cv_val = self.y.iloc[validation_indices]

            estimator = self.estimator_class(**params)
            estimator.fit(X_cv_train, y_cv_train)
            evaluation_data = get_scoring_metrics(estimator, X_cv_val, y_cv_val)

            for metric in cv_eval_metrics.keys():
                cv_eval_metrics[metric] = np.append(cv_eval_metrics[metric], evaluation_data[metric])
            trial.set_user_attr('cv_eval_metrics', cv_eval_metrics)
            trial.set_user_attr('cv_scoring_metric', cv_eval_metrics[self.scoring_metric])

            print(f'CV={cv_i} | {self.scoring_metric} {evaluation_data[self.scoring_metric]:0.3f} | rolling mean {np.mean(cv_eval_metrics[self.scoring_metric]):0.3f}')

            return evaluation_data

        n_jobs_parallel = 1
        if not 'n_jobs' in params:
            n_jobs_parallel = 8
            print('Running CV in parallel')
        
        res = Parallel(n_jobs=n_jobs_parallel, verbose=100)(delayed(run_cv)(cv_i, indices) for cv_i, indices in enumerate(self.kf.split(self.X, self.y)))


        for cv_i in range(len(res)):
            result = res[cv_i]
            for metric in result.keys():
                cv_eval_metrics[metric] = np.append(cv_eval_metrics[metric], result[metric])
            print(f'CV={cv_i} | {self.scoring_metric} {result[self.scoring_metric]:0.3f} | rolling mean {np.mean(cv_eval_metrics[self.scoring_metric]):0.3f}')
            
        trial.set_user_attr('cv_eval_metrics', cv_eval_metrics)
        trial.set_user_attr('cv_scoring_metric', cv_eval_metrics[self.scoring_metric])

        print(f'Finished trial {trial.number}')
        for metric, arr in cv_eval_metrics.items():
            print(f'{metric} {np.mean(arr):0.3f} ({np.min(arr):0.3f} - {np.max(arr):0.3f})')

        mean_scoring_metric = np.mean(cv_eval_metrics[self.scoring_metric])
        trial.set_user_attr('mean_scoring_metric', mean_scoring_metric)
        return mean_scoring_metric


### XGBoost
############################################### 

class XGBoost_OptunaObjective(OptunaObjective):
    def __init__(self, X, y, n_cv_folds=5, scoring_metric='roc_auc'):
        estimator_class = xgb.XGBClassifier
        constant_param_dict = {
            'n_jobs': -1,
            'enable_categorical': True,
            'objective': 'binary:logistic',
            'booster': 'gbtree',
            'tree_method': 'hist',
            'seed': 2024,
        }
        super().__init__(X, y, estimator_class, constant_param_dict, n_cv_folds, scoring_metric)
        
    def suggest_trial_params(self, trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12), # 3, 10
            'subsample': trial.suggest_float('subsample', 0.5, 1),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1),
            'gamma': trial.suggest_float('gamma', 1e-4, 1, log=True),
            'lambda': trial.suggest_float('lambda', 1e-3, 1000, log=True),
            #'lambda': trial.suggest_float('lambda', 1, 1000, log=True),
            'min_child_weight': trial.suggest_float('min_child_weight', 1e-4, 200, log=True), 
            'max_bin': trial.suggest_int('max_bin', 128, 1024),
        }
        return params



### LightGBM
############################################### 

class LGBM_OptunaObjective(OptunaObjective):
    def __init__(self, X, y, n_cv_folds=5, scoring_metric='roc_auc'):
        estimator_class = lgb.LGBMClassifier
        constant_param_dict = {
            'n_jobs': -1,
            'bagging_freq': 1,
            'force_row_wise': True,
            'bagging_seed': 2024,
            'verbosity': -100,
            'extra_trees': False,
        }
        super().__init__(X, y, estimator_class, constant_param_dict, n_cv_folds, scoring_metric)
        
    def suggest_trial_params(self, trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_bin': trial.suggest_int('max_bin', 128, 1024),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'num_leaves': trial.suggest_int('num_leaves', 8, 128),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-3, 1000, log=True),
            'min_sum_hessian_in_leaf': trial.suggest_float('min_sum_hessian_in_leaf', 1e-4, 10, log=True), 
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 5000, log=False),
        }
        return params



### Random Forest
############################################### 


class RF_OptunaObjective(OptunaObjective):
    def __init__(self, X, y, n_cv_folds=5, scoring_metric='roc_auc'):
        estimator_class = RandomForestClassifier
        constant_param_dict = {
            'n_jobs': -1,
            'criterion': 'gini',
            'bootstrap': True,
            'random_state': 2024,
        }
        super().__init__(X, y, estimator_class, constant_param_dict, n_cv_folds, scoring_metric)
        
    def suggest_trial_params(self, trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 5, 50),
            'max_features': trial.suggest_float('max_features', 0.025, 1),
            #'max_features': trial.suggest_int('max_features', 2, 50),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 10, 5000, log=False),
            'max_samples': trial.suggest_float('max_samples', 0.5, 1),
        }
        return params


### KNN
############################################### 


class KNN_OptunaObjective(OptunaObjective):
    def __init__(self, X, y, n_cv_folds=5, scoring_metric='roc_auc'):
        estimator_class = KNeighborsClassifier
        constant_param_dict = {
            #'n_jobs': -1,
            'algorithm': 'brute',
        }
        super().__init__(X, y, estimator_class, constant_param_dict, n_cv_folds, scoring_metric)
        
    def suggest_trial_params(self, trial):
        params = {
            'n_neighbors': trial.suggest_int('n_neighbors', 5, 499, step=2),
            'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
            'metric': trial.suggest_categorical('metric', ['manhattan', 'euclidean', 'cosine']),
        }
        return params
        

### SVC
############################################### 

class SVC_OptunaObjective(OptunaObjective):
    def __init__(self, X, y, n_cv_folds=5, scoring_metric='roc_auc'):
        estimator_class = SVC
        constant_param_dict = {
            'random_state': 2024,
            'cache_size': 7000,
            'max_iter': 1000,
        }
        super().__init__(X, y, estimator_class, constant_param_dict, n_cv_folds, scoring_metric)

    
    def suggest_trial_params(self, trial):
        params = {
            'C': trial.suggest_float('C', 1e-6, 10, log=True), # lower C = higher reg
            'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly']),
        }
        if params['kernel'] == 'rbf' or params['kernel'] == 'poly':
            params['gamma'] = trial.suggest_float('gamma', 1e-6, 10, log=True)
        if params['kernel'] == 'poly':
            params['degree'] = trial.suggest_int('degree', 2, 3)
            #params['degree'] = 3
            
        return params



### Logistic regression (with SGD)
############################################### 


class LogisticRegressionSGD_OptunaObjective(OptunaObjective):
    def __init__(self, X, y, n_cv_folds=5, scoring_metric='roc_auc'):
        estimator_class = SGDClassifier
        constant_param_dict = {
            'loss': 'log_loss',
            'penalty': 'l2',
            'fit_intercept': True,
            'tol': 1e-4,
            'n_iter_no_change': 5,
        }
        super().__init__(X, y, estimator_class, constant_param_dict, n_cv_folds, scoring_metric)
        
    def suggest_trial_params(self, trial):
        params = {
            'alpha': trial.suggest_float('alpha', 1e-6, 10, log=True), # higer alpha = higher reg
            'learning_rate': trial.suggest_categorical('learning_rate', ['optimal', 'constant', 'adaptive']),
        }

        if params['learning_rate'] != 'optimal':
            params['eta0'] = trial.suggest_float('eta0', 1e-3, 0.1, log=True)
        
        return params






# Class imbalance: oversampling / undersampling
############################################### 

class OptunaObjectiveImbalanced:
    def __init__(self, X, y, estimator_class, constant_param_dict, n_cv_folds, scoring_metric, sampler):
        self.X = X
        self.y = y
        self.estimator_class = estimator_class
        self.constant_param_dict = constant_param_dict
        self.n_cv_folds = n_cv_folds
        self.scoring_metric = scoring_metric
        self.kf = StratifiedKFold(n_splits=n_cv_folds, shuffle=True, random_state=2024)

        self.sampler = sampler

    def suggest_trial_params(self, trial):
        pass
    
    def __call__(self, trial):
        params = self.suggest_trial_params(trial)
        params.update(self.constant_param_dict)

        sampler_params = self.suggest_sampler_params(trial)

        if trial.number == 0:
            print(trial.distributions)

        print(f'\nStarting trial {trial.number}')
        print(params)

        print(sampler_params)

        trial.set_user_attr('all_params', params)

        cv_eval_metrics = {
            'roc_auc': np.array([]),
            'optimal_threshold': np.array([]),
            'sensitivity_recall': np.array([]),
            'specificity': np.array([]),
            'ppv_precision': np.array([]),
            'npv': np.array([]),
            'f1_score': np.array([]),
            'pr_auc': np.array([]),
        }

        def run_cv(cv_i, indices):
            train_indices, validation_indices = indices
            X_cv_train = self.X.iloc[train_indices]
            y_cv_train  = self.y.iloc[train_indices]
            X_cv_val = self.X.iloc[validation_indices]
            y_cv_val = self.y.iloc[validation_indices]

            #patch_sklearn()
            sampler_model = self.sampler(**sampler_params)
            X_cv_train, y_cv_train = sampler_model.fit_resample(X_cv_train, y_cv_train)

            estimator = self.estimator_class(**params)
            estimator.fit(X_cv_train, y_cv_train)
            evaluation_data = get_scoring_metrics(estimator, X_cv_val, y_cv_val)

            for metric in cv_eval_metrics.keys():
                cv_eval_metrics[metric] = np.append(cv_eval_metrics[metric], evaluation_data[metric])
            trial.set_user_attr('cv_eval_metrics', cv_eval_metrics)
            trial.set_user_attr('cv_scoring_metric', cv_eval_metrics[self.scoring_metric])

            print(f'CV={cv_i} | {self.scoring_metric} {evaluation_data[self.scoring_metric]:0.3f} | rolling mean {np.mean(cv_eval_metrics[self.scoring_metric]):0.3f}')

            return evaluation_data

        n_jobs_parallel = 1
        if not 'n_jobs' in params:
            n_jobs_parallel = 8
            print('Running CV in parallel')
        
        res = Parallel(n_jobs=n_jobs_parallel, verbose=100)(delayed(run_cv)(cv_i, indices) for cv_i, indices in enumerate(self.kf.split(self.X, self.y)))


        for cv_i in range(len(res)):
            result = res[cv_i]
            for metric in result.keys():
                cv_eval_metrics[metric] = np.append(cv_eval_metrics[metric], result[metric])
            print(f'CV={cv_i} | {self.scoring_metric} {result[self.scoring_metric]:0.3f} | rolling mean {np.mean(cv_eval_metrics[self.scoring_metric]):0.3f}')
            
        trial.set_user_attr('cv_eval_metrics', cv_eval_metrics)
        trial.set_user_attr('cv_scoring_metric', cv_eval_metrics[self.scoring_metric])

        print(f'Finished trial {trial.number}')
        for metric, arr in cv_eval_metrics.items():
            print(f'{metric} {np.mean(arr):0.3f} ({np.min(arr):0.3f} - {np.max(arr):0.3f})')

        mean_scoring_metric = np.mean(cv_eval_metrics[self.scoring_metric])
        trial.set_user_attr('mean_scoring_metric', mean_scoring_metric)
        return mean_scoring_metric


### LightGBM imbalanced
############################################### 

class LGBM_OptunaObjective_SMOTE(OptunaObjectiveImbalanced):
    def __init__(self, X, y, n_cv_folds, scoring_metric, sampler):
        estimator_class = lgb.LGBMClassifier
        constant_param_dict = {
            'n_jobs': -1,
            'bagging_freq': 1,
            'force_row_wise': True,
            'bagging_seed': 2024,
            'verbosity': -100,
            'extra_trees': False,
        }
        super().__init__(X, y, estimator_class, constant_param_dict, n_cv_folds, scoring_metric, sampler)
        
    def suggest_trial_params(self, trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_bin': trial.suggest_int('max_bin', 128, 1024),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'num_leaves': trial.suggest_int('num_leaves', 8, 128),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-3, 1000, log=True),
            'min_sum_hessian_in_leaf': trial.suggest_float('min_sum_hessian_in_leaf', 1e-4, 10, log=True), 
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 5000, log=False),
        }
        return params

    def suggest_sampler_params(self, trial):
        return {
            'sampling_strategy': 0.333, # 3:1
            'random_state': 2024,
            'k_neighbors': trial.suggest_int('k_neighbors', 5, 499, step=2),
        }

class LGBM_OptunaObjective_ADASYN(OptunaObjectiveImbalanced):
    def __init__(self, X, y, n_cv_folds, scoring_metric, sampler):
        estimator_class = lgb.LGBMClassifier
        constant_param_dict = {
            'n_jobs': -1,
            'bagging_freq': 1,
            'force_row_wise': True,
            'bagging_seed': 2024,
            'verbosity': -100,
            'extra_trees': False,
        }
        super().__init__(X, y, estimator_class, constant_param_dict, n_cv_folds, scoring_metric, sampler)
        
    def suggest_trial_params(self, trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_bin': trial.suggest_int('max_bin', 128, 1024),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'num_leaves': trial.suggest_int('num_leaves', 8, 128),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-3, 1000, log=True),
            'min_sum_hessian_in_leaf': trial.suggest_float('min_sum_hessian_in_leaf', 1e-4, 10, log=True), 
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 5000, log=False),
        }
        return params

    def suggest_sampler_params(self, trial):
        return {
            'sampling_strategy': 0.333, # 3:1
            'random_state': 2024,
            'n_neighbors': trial.suggest_int('k_neighbors', 5, 499, step=2),
        }


class LGBM_OptunaObjective_RandomOverUnderSampler(OptunaObjectiveImbalanced):
    def __init__(self, X, y, n_cv_folds, scoring_metric, sampler):
        estimator_class = lgb.LGBMClassifier
        constant_param_dict = {
            'n_jobs': -1,
            'bagging_freq': 1,
            'force_row_wise': True,
            'bagging_seed': 2024,
            'verbosity': -100,
            'extra_trees': False,
        }
        super().__init__(X, y, estimator_class, constant_param_dict, n_cv_folds, scoring_metric, sampler)
        
    def suggest_trial_params(self, trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_bin': trial.suggest_int('max_bin', 128, 1024),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'num_leaves': trial.suggest_int('num_leaves', 8, 128),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-3, 1000, log=True),
            'min_sum_hessian_in_leaf': trial.suggest_float('min_sum_hessian_in_leaf', 1e-4, 10, log=True), 
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 5000, log=False),
        }
        return params

    def suggest_sampler_params(self, trial):
        return {
            'sampling_strategy': 0.333, # 3:1
            'random_state': 2024,
            # shrinkage / smoothed bootstrap?
        }

class LGBM_OptunaObjective_ENN(OptunaObjectiveImbalanced):
    def __init__(self, X, y, n_cv_folds, scoring_metric, sampler):
        estimator_class = lgb.LGBMClassifier
        constant_param_dict = {
            'n_jobs': -1,
            'bagging_freq': 1,
            'force_row_wise': True,
            'bagging_seed': 2024,
            'verbosity': -100,
            'extra_trees': False,
        }
        super().__init__(X, y, estimator_class, constant_param_dict, n_cv_folds, scoring_metric, sampler)
        
    def suggest_trial_params(self, trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_bin': trial.suggest_int('max_bin', 128, 1024),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'num_leaves': trial.suggest_int('num_leaves', 8, 128),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-3, 1000, log=True),
            'min_sum_hessian_in_leaf': trial.suggest_float('min_sum_hessian_in_leaf', 1e-4, 10, log=True), 
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 5000, log=False),
        }
        return params

    def suggest_sampler_params(self, trial):
        return {
            'n_neighbors': trial.suggest_int('n_neighbors', 3, 99, step=2),
            'kind_sel': trial.suggest_categorical('kind_sel', ['all', 'mode']),
            'sampling_strategy': 'majority',
        }



class LGBM_OptunaObjective_SMOTE_ENN(OptunaObjectiveImbalanced):
    def __init__(self, X, y, n_cv_folds, scoring_metric, sampler):
        estimator_class = lgb.LGBMClassifier
        constant_param_dict = {
            'n_jobs': -1,
            'bagging_freq': 1,
            'force_row_wise': True,
            'bagging_seed': 2024,
            'verbosity': -100,
            'extra_trees': False,
        }
        super().__init__(X, y, estimator_class, constant_param_dict, n_cv_folds, scoring_metric, sampler)
        
    def suggest_trial_params(self, trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_bin': trial.suggest_int('max_bin', 128, 1024),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'num_leaves': trial.suggest_int('num_leaves', 8, 128),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-3, 1000, log=True),
            'min_sum_hessian_in_leaf': trial.suggest_float('min_sum_hessian_in_leaf', 1e-4, 10, log=True), 
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 5000, log=False),
        }
        return params

    def suggest_sampler_params(self, trial):
        k_neighbors_smote = trial.suggest_int('k_neighbors_smote', 5, 499, step=2)
        n_neighbors_enn = trial.suggest_int('n_neighbors_enn', 3, 15, step=2)
        kind_sel_enn = trial.suggest_categorical('kind_sel', ['all', 'mode'])
        
        return {
            'sampling_strategy': 0.333, # 3:1
            'random_state': 2024,

            'smote': SMOTE(k_neighbors=k_neighbors_smote, sampling_strategy=0.333, random_state=2024),
            'enn': EditedNearestNeighbours(n_neighbors=n_neighbors_enn, kind_sel=kind_sel_enn, sampling_strategy='majority'),
        }







################ LOGISTIC REGRESSION SMOTE ################# (3 & 10 year tte)

class LogisticRegression_OptunaObjective_SMOTE(OptunaObjectiveImbalanced):
    def __init__(self, X, y, n_cv_folds, scoring_metric, sampler):
        estimator_class = LogisticRegression
        constant_param_dict = {
            'n_jobs': -1,
            'random_state': 2024,
            'max_iter': 1000,
            'class_weight': 'balanced'
        }
        super().__init__(X, y, estimator_class, constant_param_dict, n_cv_folds, scoring_metric, sampler)
        
    def suggest_trial_params(self, trial):
        params = {
            'penalty': trial.suggest_categorical('penalty', ['l2', 'l1']),
            'C': trial.suggest_float('C', 1e-4, 100, log=True),
            'solver': trial.suggest_categorical('solver', ['lbfgs', 'liblinear', 'saga']),
            'tol': trial.suggest_float('tol', 1e-5, 1e-2, log=True),
            'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
        }
        return params

    def suggest_sampler_params(self, trial):
        return {
            'sampling_strategy': 0.333,
            'random_state': 2024,
            'k_neighbors': trial.suggest_int('k_neighbors', 3, 499, step=2),
        }

class LogisticRegression_OptunaObjective_ADASYN(OptunaObjectiveImbalanced):
    def __init__(self, X, y, n_cv_folds, scoring_metric, sampler):
        estimator_class = LogisticRegression
        constant_param_dict = {
            'n_jobs': -1,
            'random_state': 2024,
            'max_iter': 1000,
            'class_weight': 'balanced'
        }
        super().__init__(X, y, estimator_class, constant_param_dict, n_cv_folds, scoring_metric, sampler)
        
    def suggest_trial_params(self, trial):
        params = {
            'penalty': trial.suggest_categorical('penalty', ['l2', 'l1']),
            'C': trial.suggest_float('C', 1e-4, 100, log=True),
            'solver': trial.suggest_categorical('solver', ['lbfgs', 'liblinear', 'saga']),
            'tol': trial.suggest_float('tol', 1e-5, 1e-2, log=True),
            'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
        }
        return params

    def suggest_sampler_params(self, trial):
        return {
            'sampling_strategy': 0.333,
            'random_state': 2024,
            'n_neighbors': trial.suggest_int('n_neighbors', 5, 499, step=2),
        }

class LogisticRegression_OptunaObjective_RandomOverUnderSampler(OptunaObjectiveImbalanced):
    def __init__(self, X, y, n_cv_folds, scoring_metric, sampler):
        estimator_class = LogisticRegression
        constant_param_dict = {
            'n_jobs': -1,
            'random_state': 2024,
            'max_iter': 1000,
            'class_weight': 'balanced'
        }
        super().__init__(X, y, estimator_class, constant_param_dict, n_cv_folds, scoring_metric, sampler)
        
    def suggest_trial_params(self, trial):
        # First suggest penalty and solver together as a combined choice
        penalty_solver_combinations = [
            ('l1', 'liblinear'),
            ('l1', 'saga'),
            ('l2', 'lbfgs'),
            ('l2', 'liblinear'),
            ('l2', 'saga')]
        
        penalty, solver = trial.suggest_categorical('penalty_solver', penalty_solver_combinations)
        
        params = {
            'penalty': penalty,
            'solver': solver,
            'C': trial.suggest_float('C', 1e-4, 100, log=True),
            'tol': trial.suggest_float('tol', 1e-5, 1e-2, log=True),
            'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
        }
        
        return params

    def suggest_sampler_params(self, trial):
        return {
            'sampling_strategy': 0.333,
            'random_state': 2024,
        }

class LogisticRegression_OptunaObjective_ENN(OptunaObjectiveImbalanced):
    def __init__(self, X, y, n_cv_folds, scoring_metric, sampler):
        estimator_class = LogisticRegression
        constant_param_dict = {
            'n_jobs': -1,
            'random_state': 2024,
            'max_iter': 1000,
            'class_weight': 'balanced'
        }
        super().__init__(X, y, estimator_class, constant_param_dict, n_cv_folds, scoring_metric, sampler)
        
    def suggest_trial_params(self, trial):
        params = {
            'penalty': trial.suggest_categorical('penalty', ['l2', 'l1']),
            'C': trial.suggest_float('C', 1e-4, 100, log=True),
            'solver': trial.suggest_categorical('solver', ['lbfgs', 'liblinear', 'saga']),
            'tol': trial.suggest_float('tol', 1e-5, 1e-2, log=True),
            'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
        }
        return params

    def suggest_sampler_params(self, trial):
        return {
            'n_neighbors': trial.suggest_int('n_neighbors', 3, 99, step=2),
            'kind_sel': trial.suggest_categorical('kind_sel', ['all', 'mode']),
            'sampling_strategy': 'majority',
        }

class LogisticRegression_OptunaObjective_SMOTE_ENN(OptunaObjectiveImbalanced):
    def __init__(self, X, y, n_cv_folds, scoring_metric, sampler):
        estimator_class = LogisticRegression
        constant_param_dict = {
            'n_jobs': -1,
            'random_state': 2024,
            'max_iter': 1000,
            'class_weight': 'balanced'
        }
        super().__init__(X, y, estimator_class, constant_param_dict, n_cv_folds, scoring_metric, sampler)
        
    def suggest_trial_params(self, trial):
        params = {
            'penalty': trial.suggest_categorical('penalty', ['l2', 'l1']),
            'C': trial.suggest_float('C', 1e-4, 100, log=True),
            'solver': trial.suggest_categorical('solver', ['lbfgs', 'liblinear', 'saga']),
            'tol': trial.suggest_float('tol', 1e-5, 1e-2, log=True),
            'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
        }
        return params

    def suggest_sampler_params(self, trial):
        k_neighbors_smote = trial.suggest_int('k_neighbors_smote', 5, 499, step=2)
        n_neighbors_enn = trial.suggest_int('n_neighbors_enn', 3, 15, step=2)
        kind_sel_enn = trial.suggest_categorical('kind_sel', ['all', 'mode'])
        
        return {
            'sampling_strategy': 0.333,
            'random_state': 2024,
            'smote': SMOTE(k_neighbors=k_neighbors_smote, sampling_strategy=0.333, random_state=2024),
            'enn': EditedNearestNeighbours(n_neighbors=n_neighbors_enn, kind_sel=kind_sel_enn, sampling_strategy='majority'),
        }













############## XGBOOST SMOTE (5 year tte) ###################

class XGBoost_OptunaObjective_SMOTE(OptunaObjectiveImbalanced):
    def __init__(self, X, y, n_cv_folds, scoring_metric, sampler):
        estimator_class = xgb.XGBClassifier
        constant_param_dict = {
            'n_jobs': -1,
            'random_state': 2024,
            'verbosity': 0,
            'enable_categorical': True,
            'use_label_encoder': False,
            'tree_method': 'hist'
        }
        super().__init__(X, y, estimator_class, constant_param_dict, n_cv_folds, scoring_metric, sampler)
        
    def suggest_trial_params(self, trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_bin': trial.suggest_int('max_bin', 128, 1024),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'subsample': trial.suggest_float('subsample', 0.5, 1),  # ~bagging_fraction
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1),  # ~feature_fraction
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 1000, log=True),  # ~lambda_l2
            'min_child_weight': trial.suggest_float('min_child_weight', 1e-4, 10, log=True),  # ~min_sum_hessian
            'max_cat_to_onehot': trial.suggest_int('max_cat_to_onehot', 1, 4),  # For your binary features
            'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])
        }
        return params

    def suggest_sampler_params(self, trial):
        return {
            'sampling_strategy': 0.333,
            'random_state': 2024,
            'k_neighbors': trial.suggest_int('k_neighbors', 5, 499, step=2),
        }

class XGBoost_OptunaObjective_ADASYN(OptunaObjectiveImbalanced):
    def __init__(self, X, y, n_cv_folds, scoring_metric, sampler):
        estimator_class = xgb.XGBClassifier
        constant_param_dict = {
            'n_jobs': -1,
            'random_state': 2024,
            'verbosity': 0,
            'enable_categorical': True,
            'use_label_encoder': False,
            'tree_method': 'hist'
        }
        super().__init__(X, y, estimator_class, constant_param_dict, n_cv_folds, scoring_metric, sampler)
        
    def suggest_trial_params(self, trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'subsample': trial.suggest_float('subsample', 0.5, 1),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 1000, log=True),
            'min_child_weight': trial.suggest_float('min_child_weight', 1e-4, 10, log=True),
            'max_cat_to_onehot': trial.suggest_int('max_cat_to_onehot', 1, 4),
            'gamma': trial.suggest_float('gamma', 0, 5)
        }
        return params

    def suggest_sampler_params(self, trial):
        return {
            'sampling_strategy': 0.333,
            'random_state': 2024,
            'n_neighbors': trial.suggest_int('n_neighbors', 5, 499, step=2),
        }

class XGBoost_OptunaObjective_RandomOverUnderSampler(OptunaObjectiveImbalanced):
    def __init__(self, X, y, n_cv_folds, scoring_metric, sampler):
        estimator_class = xgb.XGBClassifier
        constant_param_dict = {
            'n_jobs': -1,
            'random_state': 2024,
            'verbosity': 0,
            'enable_categorical': True,
            'use_label_encoder': False,
            'tree_method': 'hist'
        }
        super().__init__(X, y, estimator_class, constant_param_dict, n_cv_folds, scoring_metric, sampler)
        
    def suggest_trial_params(self, trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'subsample': trial.suggest_float('subsample', 0.6, 1),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 100, log=True),
            'max_cat_to_onehot': 1  # Force optimal binary splits
        }
        return params

    def suggest_sampler_params(self, trial):
        return {
            'sampling_strategy': 0.333,
            'random_state': 2024,
        }

class XGBoost_OptunaObjective_ENN(OptunaObjectiveImbalanced):
    def __init__(self, X, y, n_cv_folds, scoring_metric, sampler):
        estimator_class = xgb.XGBClassifier
        constant_param_dict = {
            'n_jobs': -1,
            'random_state': 2024,
            'verbosity': 0,
            'enable_categorical': True,
            'use_label_encoder': False,
            'tree_method': 'hist'
        }
        super().__init__(X, y, estimator_class, constant_param_dict, n_cv_folds, scoring_metric, sampler)
        
    def suggest_trial_params(self, trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.1, log=True),
            'max_leaves': trial.suggest_int('max_leaves', 8, 128),  # Alternative to num_leaves
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 1000, log=True),
            'max_cat_threshold': trial.suggest_int('max_cat_threshold', 8, 64)  # For ordinal features
        }
        return params

    def suggest_sampler_params(self, trial):
        return {
            'n_neighbors': trial.suggest_int('n_neighbors', 3, 99, step=2),
            'kind_sel': trial.suggest_categorical('kind_sel', ['all', 'mode']),
            'sampling_strategy': 'majority',
        }

class XGBoost_OptunaObjective_SMOTE_ENN(OptunaObjectiveImbalanced):
    def __init__(self, X, y, n_cv_folds, scoring_metric, sampler):
        estimator_class = xgb.XGBClassifier
        constant_param_dict = {
            'n_jobs': -1,
            'random_state': 2024,
            'verbosity': 0,
            'enable_categorical': True,
            'use_label_encoder': False,
            'tree_method': 'hist'
        }
        super().__init__(X, y, estimator_class, constant_param_dict, n_cv_folds, scoring_metric, sampler)
        
    def suggest_trial_params(self, trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'subsample': trial.suggest_float('subsample', 0.5, 1),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1),
            'max_cat_to_onehot': trial.suggest_int('max_cat_to_onehot', 1, 4),
            'monotone_constraints': trial.suggest_categorical('monotone_constraints', [None, (1, -1)])
        }
        return params

    def suggest_sampler_params(self, trial):
        k_neighbors_smote = trial.suggest_int('k_neighbors_smote', 5, 499, step=2)
        n_neighbors_enn = trial.suggest_int('n_neighbors_enn', 3, 15, step=2)
        kind_sel_enn = trial.suggest_categorical('kind_sel', ['all', 'mode'])
        
        return {
            'sampling_strategy': 0.333,
            'random_state': 2024,
            'smote': SMOTE(k_neighbors=k_neighbors_smote, sampling_strategy=0.333, random_state=2024),
            'enn': EditedNearestNeighbours(n_neighbors=n_neighbors_enn, kind_sel=kind_sel_enn, sampling_strategy='majority'),
        }


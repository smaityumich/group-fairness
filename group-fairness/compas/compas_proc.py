import os,sys
import numpy as np
import pandas as pd
from sklearn import feature_extraction
from sklearn import preprocessing
from random import seed, shuffle
from aif360.algorithms.preprocessing.reweighing import Reweighing
from aif360.datasets import CompasDataset
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions\
                import load_preproc_data_compas
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import tensorflow as tf

from sklearn.linear_model import LogisticRegressionCV
from numpy.linalg import matrix_rank
import data_pre  

def get_compas_orig():
    dataset_orig = data_pre.load_preproc_data_compas()

    return dataset_orig


def get_compas_train_test(
        dataset_orig,
        pct=0.8,
        seed=None,
        removeProt=True,
        SenSR=True,
        ):
    
    # we will standardize continous features
    continous_features = [
            'priors_count'
        ]
    continous_features_indices = [
            dataset_orig.feature_names.index(feat) 
            for feat in continous_features
        ]
    
    privileged_groups = [{'race': 1}]
    unprivileged_groups = [{'race': 0}]
    
    # Get the dataset and split into train and test
    dataset_orig_train, dataset_orig_vt = dataset_orig.split([pct-0.2], shuffle=True)
    dataset_orig_valid, dataset_orig_test = dataset_orig_vt.split([0.5], shuffle=True)
#     dataset_orig_train, dataset_orig_test = dataset_orig.split([pct], shuffle=True)
    """
    Reweighing dataset:
        1. use the splitting method of aif360 (that data structure has an attribute "instance_weights")
        2. get weight of original and reweighing data
        3. for fair boost: component-wise product of weights of  reweighing data and attacking weights
        4. Only for disjoint set, if we want both for gender and race, create sets: white+male, white+female, black+male, black+female
    """
    RW = Reweighing(unprivileged_groups=unprivileged_groups,
               privileged_groups=privileged_groups)
    RW.fit(dataset_orig_train)
    dataset_transf_train = RW.transform(dataset_orig_train)
    
    RWv = Reweighing(unprivileged_groups=unprivileged_groups,
               privileged_groups=privileged_groups)
    RWv.fit(dataset_orig_valid)
    dataset_transf_valid = RW.transform(dataset_orig_valid)
    
    RWt = Reweighing(unprivileged_groups=unprivileged_groups,
               privileged_groups=privileged_groups)
    RWt.fit(dataset_orig_test)
    dataset_transf_test = RW.transform(dataset_orig_test)
    
    
    X_train = dataset_transf_train.features
    X_test = dataset_transf_test.features
    X_valid = dataset_transf_valid.features
    y_train = dataset_transf_train.labels
    y_test = dataset_transf_test.labels
    y_valid = dataset_transf_valid.labels
    w_train = dataset_transf_train.instance_weights.ravel()
    w_test = dataset_transf_test.instance_weights.ravel()
    w_valid = dataset_transf_valid.instance_weights.ravel()
    
    y_train = np.reshape(y_train, (-1, ))
    y_test = np.reshape(y_test, (-1, ))
    y_valid = np.reshape(y_valid, (-1, ))
    
    sind = dataset_orig.feature_names.index('sex')
    rind = dataset_orig.feature_names.index('race')
    print(dataset_orig.feature_names[sind])
    print(dataset_orig.feature_names[rind])
    y_sex_train = X_train[:, sind]
    y_sex_test = X_test[:, sind]
    y_race_train = X_train[:, rind]
    y_race_test = X_test[:, rind]
    
    ### PROCESS TRAINING DATA
    # normalize continuous features
    SS = StandardScaler().fit(X_train[:, continous_features_indices])
    X_train[:, continous_features_indices] = SS.transform(
            X_train[:, continous_features_indices]
    )

    A = None
        
    ### PROCESS TEST DATA
    # normalize continuous features
    X_test[:, continous_features_indices] = SS.transform(
            X_test[:, continous_features_indices]
    )
    X_valid[:, continous_features_indices] = SS.transform(
            X_valid[:, continous_features_indices]
    )

    return X_train, X_test, y_train, y_test, y_sex_train, y_sex_test, y_race_train, y_race_test, dataset_orig.feature_names, A, w_train, w_test, X_valid, y_valid, w_valid


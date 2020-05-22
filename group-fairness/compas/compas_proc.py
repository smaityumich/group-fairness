import os,sys
import numpy as np
import pandas as pd
from sklearn import feature_extraction
from sklearn import preprocessing
from random import seed, shuffle
from sklearn.model_selection import train_test_split
from aif360.algorithms.preprocessing.reweighing import Reweighing
from aif360.datasets import CompasDataset
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions\
                import load_preproc_data_compas
from IPython.display import display, Markdown, Latex
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from copy import deepcopy
import script
import scipy as sp
import tensorflow as tf
import xgboost as xgb

from sklearn.linear_model import LogisticRegressionCV
from numpy.linalg import matrix_rank
import data_pre  

def get_compas_orig():
    dataset_orig = data_pre.load_preproc_data_compas()
    display(Markdown("#### Dataset shape"))
    print(dataset_orig.features.shape)
    display(Markdown("#### Dataset feature names"))
    print(dataset_orig.feature_names)
    dataset_orig.features = dataset_orig.features[:,:-1]

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

def compas_feature_logreg(
        X_train,
        X_test=None,
        rind=None,
        feature_names=None,
        test=True,
        labels=None
        ):

    if feature_names and rind:
        print("Fitting LR to feature {}".format(feature_names[rind]))

    LR = LogisticRegressionCV(Cs=100, cv=5, max_iter=5000)

    # data for training logistic regression
    if X_test is not None:
        XLR = np.vstack((X_train, X_test))
    else:
        XLR = np.copy(X_train)

    if rind is not None: 
        targets = XLR[:,rind].copy()
        XLR[:,rind] = np.zeros(XLR.shape[0])

    elif labels is not None: 
        targets = labels

    else:
        print("Error: no labels provided for logistic regression")
        print("Error: no model trained", flush=True)
        return LR

    LR.fit( XLR, targets )

    if test and X_test is not None:
        outputs = LR.predict(np.vstack((X_train,X_test)))
        print("Training error of LR classifier: {}".format(
                np.abs(outputs-targets).sum()/XLR.shape[0]
            ))

    return LR

# projection onto sensitive subspace
def compas_proj_matrix_sensr(X_train, rind, feature_names, test=False,save=False):

    eg = np.zeros(X_train.shape[1])
    eg[feature_names.index('sex')] = 1.0

    er = np.zeros(X_train.shape[1])
    er[feature_names.index('race')] = 1.0

    TLR = compas_feature_logreg(X_train, rind=rind, test=test)
    wg = TLR.coef_[0]

    A = np.array([wg, eg, er]).T

    return script.proj_matrix_gen(X_train, A, test, save)


# Import the one-hot encoded data here or not whatever
# Good: xgb.train(param, dtrain, 1000, watchlist). Gets balanced accuracy of ~84%
def compas_prep_baseline_xgb(X_train, X_test, X_valid, y_train, y_test, y_valid, w_train, w_test, w_valid, proj, reweight, flag_project):

    y_train_real = y_train
    y_test_real = y_test
    y_valid_real = y_valid
#     if (len(y_train.shape) > 1):
#         y_train_real = y_train[:,1].copy().astype('int')
#         y_test_real = y_test[:,1].copy().astype('int')
#     reweight = True
    if reweight:
        print('Reweighing the data')
        dtrain = xgb.DMatrix(data=X_train, label=y_train_real, weight=w_train)
#         dtest_rw = xgb.DMatrix(data=X_test, label=y_test_real, weight=w_test)
        dtest = xgb.DMatrix(data=X_test, label=y_test_real)
        dvalid_rw = xgb.DMatrix(data=X_valid, label=y_valid_real, weight=w_valid)
        dvalid = dvalid_rw
#         dvalid = xgb.DMatrix(data=X_valid, label=y_valid_real)
        
        watchlist = [(dtrain, 'train_rw'), (dtest, 'test'), (dvalid, 'valid_rw')]
#         dtest_pr = None
    elif flag_project:
        print('Projecting the data')
        X_train_pr = np.matmul(X_train, proj)
        X_test_pr = np.matmul(X_test, proj)
        X_valid_pr = np.matmul(X_valid, proj)
        dtrain = xgb.DMatrix(data=X_train_pr, label=y_train_real)
        dtest_pr = xgb.DMatrix(data=X_test_pr, label=y_test_real)
        dvalid_pr = xgb.DMatrix(data=X_valid_pr, label=y_valid_real)
        dvalid = dvalid_pr
        dtest = dtest_pr
        watchlist = [(dtrain, 'train_pr'), (dtest, 'test_pr'), (dvalid, 'valid_pr')]
#         dtest_rw = None
    else:
        dtrain = xgb.DMatrix(data=X_train, label=y_train_real)
        dtest = xgb.DMatrix(data=X_test, label=y_test_real)
        dvalid = xgb.DMatrix(data=X_valid, label=y_valid_real)
        watchlist = [(dtrain, 'train'), (dtest, 'test'), (dvalid, 'valid')]
        
    num1 = y_train_real.sum()
    num0 = (1-y_train_real).sum()

    return dtrain, dtest, dvalid, watchlist, num0, num1


def compas_setup(
        pct=0.8, 
        seed=None,
        removeProt=True,
        #dtype='float32', 
        #tdtype=tf.float32
        ):
    
    # pull in the compas data

    orig_dataset = get_compas_orig()
    X_train, X_test, y_train, y_test, y_sex_train, y_sex_test, y_race_train, y_race_test, feature_names, A, w_train, w_test, X_valid, y_valid, w_valid = get_compas_train_test(orig_dataset, pct=pct, seed=seed, removeProt=removeProt, SenSR=True)

    # fair metric
    if removeProt:
        print('Computing fair metric')
        sind = feature_names.index('sex')
        rind = feature_names.index('race')
        proj = compas_proj_matrix_sensr(
            X_train, rind, feature_names, test=False,save=False
            )
        projData = np.matmul(X_train, proj)

    Corig = sp.spatial.distance.squareform(
            sp.spatial.distance.pdist(
                projData,
                metric='sqeuclidean'
            ))#.astype(dtype)
    print(Corig.shape, Corig.mean())

    return X_train, X_test, y_train, y_test, y_sex_train, y_sex_test, y_race_train, y_race_test, Corig, proj, w_train, w_test, feature_names, X_valid, y_valid, w_valid


def sex_cons(bst, X_test, sind, rind):

    # Only make one copy for SPEED
    X = deepcopy(X_test)

    X[:, sind] = np.zeros(X.shape[0])
    preds0 = bst.predict(xgb.DMatrix(X))
    y0 = np.array([ 1 if pval > 0.5 else 0 for pval in preds0])

    X[:, sind] = np.ones(X.shape[0])
    preds1 = bst.predict(xgb.DMatrix(X))
    y1 = np.array([ 1 if pval > 0.5 else 0 for pval in preds1])

    # Want all quadruples to be consistent (all 0s or all 1s)
    return (y0*y1).sum() + ((1-y0)*(1-y1)).sum()

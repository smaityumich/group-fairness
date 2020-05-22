import numpy as np
import tensorflow as tf
from adult_modified import preprocess_adult_data
import setup2
import sys

# Adult data processing
seed = 1
dataset_orig_train, dataset_orig_test = preprocess_adult_data(seed = seed)

all_train, all_test = dataset_orig_train.features, dataset_orig_test.features
y_train, y_test = dataset_orig_train.labels.reshape((-1,)), dataset_orig_test.labels.reshape((-1,))
y_train, y_test = y_train.astype('int32'), y_test.astype('int32')

x_train = np.delete(all_train, [dataset_orig_test.feature_names.index(feat) for feat in ['sex_ Male', 'race_ White']], axis = 1)
x_test = np.delete(all_test, [dataset_orig_test.feature_names.index(feat) for feat in ['sex_ Male', 'race_ White']], axis = 1)

group_train = dataset_orig_train.features[:, [dataset_orig_test.feature_names.index(feat) for feat in ['sex_ Male', 'race_ White']]]
group_test = dataset_orig_test.features[:, [dataset_orig_test.feature_names.index(feat) for feat in ['sex_ Male', 'race_ White']]]

x_train, y_train, group_train = tf.cast(x_train, dtype = tf.float32),\
     tf.one_hot(y_train, 2), tf.cast(group_train, dtype = tf.int32)


x_test, y_test, group_test = tf.cast(x_test, dtype = tf.float32),\
     tf.one_hot(y_test, 2), tf.cast(group_test, dtype = tf.int32)

data_train = x_train, y_train, group_train
data_test = x_test, y_test, group_test

import itertools
groups = list(itertools.product([0,1], [0,1]))
groups = [list(x) for x in groups]
groups = tf.cast(groups, tf.int32)

grouped_x_train = [x_train[tf.reduce_all(group_train == g, axis = 1)] for g in groups]
grouped_x_test = [x_test[tf.reduce_all(group_test == g, axis = 1)] for g in groups]
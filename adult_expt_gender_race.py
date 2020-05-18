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

# Experiment
batch_size = 400
epoch = 8000
l2_regularizer = 0
lr = 1e-4
wlr = float(sys.argv[2])
w_reg = float(sys.argv[1])
epsilon = float(sys.argv[3])
start_training = 250

seed = np.random.randint(10000)

experiment = setup2.GroupFairness(batch_size=batch_size, epoch=epoch, l2_regularizer= l2_regularizer,\
     learning_rate=lr, wasserstein_lr=wlr, wasserstein_regularizer=w_reg, seed=seed, epsilon=epsilon, start_training=start_training)
experiment.set_graph(classifier_input_shape=(39,), n_groups=4, \
    classifier_architecture=[50, 50, 10, 2], potential_architecture=[50, 10, 1])
experiment.fit(data_train, data_test, groups, group_names=['gender', 'race'])

import numpy as np
import tensorflow as tf
from compas_proc import get_compas_train_test, get_compas_orig
import setup2
import sys
import tensorflow as tf

# Adult data processing
seed = 1
X_train, X_test, y_train, y_test, _, _, _, _, _, \
     _, _, _, _ , _, _ = get_compas_train_test(get_compas_orig(), seed=seed)
x_train, group_train = X_train[:, 2:], X_train[:, :2]
x_test, group_test = X_test[:, 2:], X_test[:, 2:]


x_train, y_train, group_train = tf.cast(x_train, dtype = tf.float32),\
     tf.one_hot(y_train.astype('int32'), 2), tf.cast(group_train, dtype = tf.int32)


x_test, y_test, group_test = tf.cast(x_test, dtype = tf.float32),\
     tf.one_hot(y_test.astype('int32'), 2), tf.cast(group_test, dtype = tf.int32)

data_train = x_train, y_train, group_train
data_test = x_test, y_test, group_test

import itertools
groups = list(itertools.product([0,1], [0,1]))
groups = [list(x) for x in groups]
groups = tf.cast(groups, tf.int32)

# Experiment
batch_size = 400
epoch = 1000
l2_regularizer = 0
lr = 1e-4
wlr = 1e-4#float(sys.argv[2])
w_reg = 20#float(sys.argv[1])
epsilon = 0.1#float(sys.argv[3])
start_training = 250

seed = 1

x_dim = x_train.shape[1]

experiment = setup2.GroupFairness(batch_size=batch_size, epoch=epoch, l2_regularizer= l2_regularizer,\
     learning_rate=lr, wasserstein_lr=wlr, wasserstein_regularizer=w_reg, seed=seed, epsilon=epsilon, start_training=start_training)
experiment.set_graph(classifier_input_shape=(x_dim,), n_groups=4, \
    classifier_architecture=[100, 100, 50, 2], potential_architecture=[50, 20, 1])
experiment.fit(data_train, data_test, groups, group_names=['gender', 'race'])

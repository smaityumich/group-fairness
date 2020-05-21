import numpy as np
import tensorflow as tf
from adult_modified import preprocess_adult_data
import setup
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
     tf.cast(y_train, dtype = tf.float32), tf.cast(group_train, dtype = tf.int32)


x_test, y_test, group_test = tf.cast(x_test, dtype = tf.float32),\
     tf.cast(y_test, dtype = tf.float32), tf.cast(group_test, dtype = tf.int32)

data_train = x_train, y_train, group_train
data_test = x_test, y_test, group_test

import itertools
groups = list(itertools.product([0,1], [0,1]))
groups = [list(x) for x in groups]
groups = tf.cast(groups, tf.int32)

# Experiment
alpha, beta = float(sys.argv[1]), float(sys.argv[2])
learning_rate = float(sys.argv[3])
if len(sys.argv) <= 4:
     train_batch_size = 1000
else:
     train_batch_size = int(float(sys.argv[4]))

epoch = 10000


seed = 1

# Setting up unit experiment
expt = setup.unit_expt(train_batch_size=train_batch_size, alpha=alpha, beta=beta,\
      seed= seed, learning_rate=learning_rate, epoch = epoch)
expt.fit(data_train, data_test)


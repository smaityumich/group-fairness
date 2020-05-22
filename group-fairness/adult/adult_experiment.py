import numpy as np
import tensorflow as tf
from adult_modified import preprocess_adult_data
import setup

# Adult data processing
seed = 1
dataset_orig_train, dataset_orig_test = preprocess_adult_data(seed = seed)

x_train, x_test = dataset_orig_train.features, dataset_orig_test.features
y_train, y_test = dataset_orig_train.labels.reshape((-1,)), dataset_orig_test.labels.reshape((-1,))
y_train, y_test = y_train.astype('int32'), y_test.astype('int32')

x_gender_train = np.delete(x_train, [dataset_orig_test.feature_names.index(feat) for feat in ['sex_ Male']], axis = 1)
x_gender_test = np.delete(x_test, [dataset_orig_test.feature_names.index(feat) for feat in ['sex_ Male']], axis = 1)

y_gender_train = dataset_orig_train.features[:, dataset_orig_train.feature_names.index('sex_ Male')]
y_gender_test = dataset_orig_test.features[:, dataset_orig_test.feature_names.index('sex_ Male')]

x_gender_train, y_gender_train, y_train = tf.cast(x_gender_train, dtype = tf.float32),\
     tf.cast(y_gender_train, dtype = tf.int32), tf.one_hot(y_train, 2)


x_gender_test, y_gender_test, y_test = tf.cast(x_gender_test, dtype = tf.float32),\
     tf.cast(y_gender_test, dtype = tf.int32), tf.one_hot(y_test, 2)

data_train = x_gender_train, y_train, y_gender_train
data_test = x_gender_test, y_test, y_gender_test

# Experiment
batch_size = 500
epoch = 6000
l2_regularizer = 0.1
lr = 5e-4
wlr = 5e-4
w_reg = 10



experiment = setup.GroupFairness(batch_size=batch_size, epoch=epoch, l2_regularizer= l2_regularizer,\
     learning_rate=lr, wasserstein_lr=wlr, wasserstein_regularizer=w_reg)
experiment.set_graph()
experiment.fit(data_train, data_test)

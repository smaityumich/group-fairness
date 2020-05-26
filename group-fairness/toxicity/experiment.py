import io
import os
import shutil
import sys
import tempfile
import zipfile
from urllib import request

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing import text
import setup



hparams = {
    "cnn_filter_sizes": [128, 128, 128],
    "cnn_kernel_sizes": [5, 5, 5],
    "cnn_pooling_sizes": [5, 5, 40],
    "embedding_dim": 100,
    "embedding_trainable": False,
    "learning_rate": 0.005,
    "max_num_words": 10000,
    "max_sequence_length": 250
}



data_train = pd.read_csv('~/toxicity-data/wiki_train.csv')
data_test = pd.read_csv('~/toxicity-data/wiki_test.csv')
data_vali = pd.read_csv('~/toxicity-data/wiki_vali.csv')

labels_train = data_train["is_toxic"].values.reshape(-1, 1) * 1.0
labels_test = data_test["is_toxic"].values.reshape(-1, 1) * 1.0
labels_vali = data_vali["is_toxic"].values.reshape(-1, 1) * 1.0

tokenizer = text.Tokenizer(num_words=hparams["max_num_words"])
tokenizer.fit_on_texts(data_train["comment"])

def prep_text(texts, tokenizer, max_sequence_length):
    # Turns text into into padded sequences.
    text_sequences = tokenizer.texts_to_sequences(texts)
    return sequence.pad_sequences(text_sequences, maxlen=max_sequence_length)

text_train = prep_text(data_train["comment"], tokenizer, hparams["max_sequence_length"])
text_test = prep_text(data_test["comment"], tokenizer, hparams["max_sequence_length"])
text_vali = prep_text(data_vali["comment"], tokenizer, hparams["max_sequence_length"])

terms = {
    'sexuality': ['gay', 'lesbian', 'bisexual', 'homosexual', 'straight', 'heterosexual'],
    'gender identity': ['trans', 'transgender', 'cis', 'nonbinary'],
    'religion': ['christian', 'muslim', 'jewish', 'buddhist', 'catholic', 'protestant', 'sikh', 'taoist'],
    'race': ['african', 'african american', 'black', 'white', 'european', 'hispanic', 'latino', 'latina',
             'latinx', 'mexican', 'canadian', 'american', 'asian', 'indian', 'middle eastern', 'chinese',
             'japanese']}

group_names = list(terms.keys())
num_groups = len(group_names)


def get_groups(text):
    # Returns a boolean NumPy array of shape (n, k), where n is the number of comments,
    # and k is the number of groups. Each entry (i, j) indicates if the i-th comment
    # contains a term from the j-th group.
    groups = np.zeros((text.shape[0], num_groups))
    for ii in range(num_groups):
        groups[:, ii] = text.str.contains('|'.join(terms[group_names[ii]]), case=False)
    return groups

groups_train = get_groups(data_train["comment"])
groups_test = get_groups(data_test["comment"])
groups_vali = get_groups(data_vali["comment"])


embedding_matrix =np.load('embedding-matrix.npy')

# Setting up data
text_train, text_test = tf.cast(text_train, dtype = tf.float32), tf.cast(text_test, dtype = tf.float32)
labels_train, labels_test = tf.cast(labels_train, dtype=tf.int32), tf.cast(labels_test, dtype = tf.int32)
labels_train, labels_test = tf.reshape(labels_train, [-1,]), tf.reshape(labels_test, [-1])
labels_train, labels_test = tf.one_hot(labels_train, 2), tf.one_hot(labels_test, 2)
groups_train, groups_test  = tf.cast(groups_train, dtype = tf.int8), tf.cast(groups_test, dtype = tf.int8)
data_train = text_train, labels_train, groups_train
data_test = text_test, labels_test, groups_test

# Setup parameters 
lr = 0.001#float(sys.argv[1])
wlr = 0.001#float(sys.argv[2])
w_reg = 10#float(sys.argv[3])
eps = 0.1#float(sys.argv[4])
epoch = 10
start_training = 5
seed = 1

experiment = setup.GroupFairness(learning_rate=lr, wasserstein_lr=wlr, wasserstein_regularizer=w_reg,\
 epoch=epoch, seed=seed, start_training=start_training, epsilon=eps)

experiment.protected_groups(None)
experiment.set_graph(embedding_matrix, potential_architecture=[10, 10, 1])
experiment.fit(data_train, data_test)

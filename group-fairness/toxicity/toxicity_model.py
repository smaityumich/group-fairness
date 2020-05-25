import tensorflow as tf
from tensorflow import keras
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


from tensorflow import keras







def toxicity_model(embedding_matrix, hparams = {\
    "cnn_filter_sizes": [128, 128, 128],\
    "cnn_kernel_sizes": [5, 5, 5],\
    "cnn_pooling_sizes": [5, 5, 40],\
    "embedding_dim": 100,\
    "embedding_trainable": False,\
    "max_sequence_length": 250\
}\
):
    model = keras.Sequential()

    # Embedding layer.
    embedding_layer = keras.layers.Embedding(
        embedding_matrix.shape[0], embedding_matrix.shape[1], weights=[embedding_matrix],
        input_length=hparams["max_sequence_length"], trainable=hparams['embedding_trainable'])
    model.add(embedding_layer)

    # Convolution layers.
    for filter_size, kernel_size, pool_size in zip(
        hparams['cnn_filter_sizes'], hparams['cnn_kernel_sizes'], hparams['cnn_pooling_sizes']):

        conv_layer = keras.layers.Conv1D(filter_size, kernel_size, activation='relu', padding='same')
        model.add(conv_layer)

        pooled_layer = keras.layers.MaxPooling1D(pool_size, padding='same')
        model.add(pooled_layer)

    # Add a flatten layer, a fully-connected layer and an output layer.
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(2))

    return model





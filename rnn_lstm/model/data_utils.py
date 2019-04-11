import os
import numpy as np
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


def get_data(num_words=5000, max_sequence_length= 500):
    """
    Loads and formats the imdb dataset.

    :param num_words: int; size of the vocabulary
    :param max_sequence_length: int; maximum length of sequences
    :return: tuple; X_train, X_test, y_train, y_test
    """

    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=num_words)
    X_train = pad_sequences(X_train, maxlen = max_sequence_length)
    X_test = pad_sequences(X_test, maxlen = max_sequence_length)
    return X_train, X_test, y_train, y_test

def get_minibatches(X, y, batch_size=128, categorical=True):
    """
    Organize the data into minibatches.

    :param X: array; should have shape [num_samples, max_sequence_length]
    :param y: array; should have shape [num_sampes, num_classes]
    :param batch_size: int; number of samples in a match
    :param categorical: Boolean; if True, converts y to categorical
    :return: list of tuples; minibatches of the form (X_batch, y_batch)
    """

    if categorical:
        y = to_categorical(y)

    minibatches = []
    X_batch = np.zeros([batch_size, X.shape[1]])
    y_batch = np.zeros([batch_size, y.shape[1]])
    idx = 0
    for x_, y_ in zip(X, y):
        X_batch[idx] = x_
        y_batch[idx] = y_
        idx += 1
        if idx >= batch_size:
            minibatches.append((X_batch, y_batch))
            idx = 0
    return minibatches

import numpy as np
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, f1_score


def get_data(num_words=5000, max_sequence_length=500, y_as_column=False):
    """
    Loads and formats the imdb dataset.

    :param num_words: int; size of the vocabulary
    :param max_sequence_length: int; maximum length of sequences
    :param y_as_column: Boolean; if try, y is converted to a column vector [n_samples, 1]
    This should be set to True for the tensorflow model
    :return: tuple; X_train, X_test, y_train, y_test
    """

    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=num_words)
    X_train = pad_sequences(X_train, maxlen=max_sequence_length)
    X_test = pad_sequences(X_test, maxlen=max_sequence_length)
    if y_as_column:
        y_train = np.array(y_train).reshape(-1, 1)
        y_test = np.array(y_train).reshape(-1, 1)
    return X_train, X_test, y_train, y_test

def assess(predicted, actual):
    """
    Get accuracy metrics.

    :param predicted: np.array; predicted labels
    :param actual: np.array; actual labels
    :return: dict; performace metrics (accuracy and f1)
    """

    predicted = np.where(predicted > 0.5, 1, 0)
    accuracy = accuracy_score(predicted, actual)
    f1 = f1_score(predicted, actual)
    return {
        "accuracy": accuracy,
        "f1": f1
    }
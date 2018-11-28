from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences

def get_data(num_words = 5000, max_sequence_length = 500):
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=num_words)
    X_train = pad_sequences(X_train, maxlen = max_sequence_length)
    X_test = pad_sequences(X_test, maxlen = max_sequence_length)
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_data()
    print(X_train.shape)
    print(y_train.shape)



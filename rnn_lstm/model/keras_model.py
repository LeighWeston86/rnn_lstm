from keras.layers import Dense, LSTM
from keras.layers.embeddings import Embedding
from keras.models import Input, Model
from keras.optimizers import Adam

class KerasLSTM(object):

    def __init__(self, sequence_length=500, embedding_dim=32, vocab_size=5000, lstm_size=100, dropout=0.4):
        """
        A simple LSTM to perform binary text classification.

        :param sequence_length: int; maximum length of sequences
        :param embedding_dim: int; dimension of the word embeddings
        :param vocab_size: int; number of words in the vocab
        :param lstm_size: int; size of the LSTM hidden layer
        :param dropout: float; dropout hyperparameter
        """

        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.lstm_size = lstm_size
        self.dropout = dropout
        self.model = None

    def lstm(self):
        """
        Defines the model architecture.

        :return: keras.models.Model; the LSTM model
        """

        #Input layer
        input_layer = Input(shape=(self.sequence_length,))

        #Embedding layer
        embedding_layer = Embedding(input_dim=self.vocab_size,
                              output_dim=self.embedding_dim,
                              input_length=self.sequence_length)

        #LSTM layer
        lstm = embedding_layer(input_layer)
        lstm = LSTM(self.lstm_size, dropout=self.dropout, recurrent_dropout=self.dropout)(lstm)

        #Output layer
        out = Dense(1, activation="sigmoid")(lstm)
        model = Model(input_layer, out)

        return model

    def fit(self, X_train, y_train, learning_rate=0.01, n_epochs=3):
        """
        Fits the model to the data.

        :param X_train: array; a two dimensional array of shape (n_samples, sequence_length)
        :param y_train: array; a one dimensional array of shape (n_samples,)
        :param learning_rate: float; learning rate for Adam optimizer
        :param n_epochs: int; number of passes through the data
        """

        self.model = self.lstm()
        adam = Adam(lr=learning_rate)
        self.model.compile(optimizer=adam, loss="binary_crossentropy", metrics=["accuracy"])
        self.model.fit(X_train, y_train, batch_size=256, epochs=n_epochs)

    def predict(self, X_test):
        """
        Makes predictions.

        :param X_test: array; a two dimensional array of shape (n_samples, sequence_length)
        :return: array; a vector of predictions
        """

        return np.where(self.model.predict(X_test) > 0.5, 1, 0)


if __name__ == "__main__":
    from sklearn.metrics import f1_score
    from rnn_lstm.model.data_utils import get_data
    import numpy as np
    X_train, X_test, y_train, y_test = get_data()
    print(X_train.shape)
    print(y_train.shape)
    lstm = KerasLSTM()
    lstm.fit(X_train, y_train)
    train_predicted = np.where(lstm.predict(X_train) > 0.5, 1, 0)
    test_predicted = np.where(lstm.predict(X_test) > 0.5, 1, 0)
    print('Train set f1: {}'.format(f1_score(train_predicted, y_train)))
    print('Test set f1: {}'.format(f1_score(test_predicted, y_test)))










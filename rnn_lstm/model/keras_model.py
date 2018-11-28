from rnn_lstm.utils.data_utils import get_data
from keras.layers import Dense, LSTM, Dropout
from keras.layers.embeddings import Embedding
from keras.models import Input, Model
from keras.optimizers import Adam

class KerasLSTM:

    def __init__(self, sequence_length = 500, embedding_dim = 32, vocab_size = 5000, lstm_size = 100, dropout = 0.2):
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.lstm_size = lstm_size
        self.dropout = dropout
        self.model = None

    def lstm(self):

        #Input layer
        input_layer = Input(shape = (self.sequence_length,))

        #Embedding layer
        embedding_layer = Embedding(input_dim = self.vocab_size,
                              output_dim = self.embedding_dim,
                              input_length = self.sequence_length)

        #LSTM model
        lstm = embedding_layer(input_layer)
        lstm = LSTM(self.lstm_size, dropout = self.dropout, recurrent_dropout = self.dropout)(lstm)

        #Output layer
        out = Dense(1, activation='sigmoid')(lstm)
        model = Model(input_layer, out)

        return model

    def fit(self, X_train, y_train, learning_rate = 0.01, n_epochs = 3):
        self.model = self.lstm()
        adam = Adam(lr = learning_rate)
        self.model.compile(optimizer = adam, loss = 'binary_crossentropy', metrics = ['accuracy'])
        self.model.fit(X_train, y_train, batch_size = 256, epochs = n_epochs)

    def predict(self, X_test):
        return self.model.predict(X_test)


if __name__ == '__main__':
    from sklearn.metrics import f1_score
    X_train, X_test, y_train, y_test = get_data()
    lstm = KerasLSTM()
    lstm.fit(X_train, y_train)
    train_predicted =  [1 if pred[0] > 0.5 else 0 for pred in lstm.predict(X_train)]
    test_predicted = [1 if pred[0] > 0.5 else 0 for pred in lstm.predict(X_test)]
    print('Train set f1: {}'.format(f1_score(train_predicted, y_train)))
    print('Test set f1: {}'.format(f1_score(test_predicted, y_test)))










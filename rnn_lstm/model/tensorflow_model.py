import math
import tensorflow as tf
import numpy as np

class TfLSTM(object):

    def __init__(self, vocab_size,
                 num_classes,
                 learning_rate=0.01,
                 lstm_size=100,
                 sequence_length=500,
                 embedding_size=32,
                 dropout=0.4):

        """
        An LSTM for text classification.

        :param vocab_size: int; number of words in the vocabulary
        :param num_classes: int; number of label classes
        :param learning_rate: float; learning rate for the Adam optimizer
        :param lstm_size: int; size of the LSTM hidden layer
        :param sequence_length: int; maximum input sequence length
        :param embedding_size: int; size of the word embeddings to be trained
        :param dropout: float; regularization of the hidden layer
        """

        # Define the instance attributes
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.lstm_size = lstm_size
        self.sequence_length = sequence_length
        self.embedding_size = embedding_size
        self.dropout = dropout

        # Build the graph
        self._add_placeholders()
        self._add_embeddings_op()
        self._add_lstm_op()
        self._add_loss_op()

        # Start the session
        self.sess = tf.Session()

    def _add_placeholders(self):
        """
        Adds the placeholders to the graph. Defines self.X, self.y and self.keep_prob.
        """
        self.X = tf.placeholder(shape=[None, self.sequence_length],
                                dtype=tf.int32, name="X")
        self.y = tf.placeholder(shape=[None, 1], dtype=tf.float32, name="y")
        self.keep_prob = tf.placeholder(shape=[], dtype=tf.float32, name="keep_prob")

    def _add_embeddings_op(self):
        """
        Creates the embeddings layer, defines self.embed
        """

        embedding = tf.Variable(tf.random_uniform(shape=[self.vocab_size, self.embedding_size], minval=-1, maxval=1))
        self.embed = tf.nn.embedding_lookup(embedding, self.X)

    def _add_lstm_op(self):
        """
        Adds the LSTM, sefines self.logits
        """

        # LSTM with dropout - inputs have shape [batch_size, sequence_length, embedding_size]
        lstm = tf.contrib.rnn.BasicLSTMCell(self.lstm_size)
        lstm = tf.contrib.rnn.DropoutWrapper(cell=lstm, output_keep_prob=1-self.keep_prob)
        h, final_states = tf.nn.dynamic_rnn(lstm, self.embed, dtype=tf.float32)
        lstm_out = final_states.h # lstm_out has shape [batch_size, lstm_size]

        # Output layer
        weights = tf.Variable(tf.truncated_normal([self.lstm_size, self.num_classes]))
        bias = tf.Variable(tf.random_normal([self.num_classes]))
        self.logits = tf.add(tf.matmul(lstm_out, weights), bias)

    def _add_loss_op(self):
        """
        Defines the cost function using softmax cross entropy and the Adam optimizer;
        defines self.cost, self.optimizer.
        """
        self.cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

    @staticmethod
    def _get_batches(X, y, batch_size):
        num_batches = math.ceil(X.shape[0] / batch_size)
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        dataset = dataset.batch(num_batches)
        iterator = dataset.make_one_shot_iterator()
        return iterator, num_batches

    def fit(self, X, y, batch_size=256, num_epochs=10, verbose=True):
        """
        Fit the model.

        :param batches: list of tuples; each tuple of the form (X_batch, y_batch)
        :param num_epochs: int; number of passes over the training set
        :param verbose: Boolean; if True, prints loss after each epoch
        """

        # Initialize
        init = tf.global_variables_initializer()
        self.sess.run(init)

        # Begin training
        for epoch in range(num_epochs):

            # New epoch
            iterator, num_batches = self._get_batches(X, y, batch_size)
            next_batch = iterator.get_next()
            epoch_loss = 0
            for batch in range(int(num_batches)):

                X_batch, y_batch = self.sess.run(next_batch)

                feed_dict = {
                    self.X: X_batch,
                    self.y: y_batch,
                    self.keep_prob: 1 - self.dropout
                }
                minibatch_cost, _ = self.sess.run([self.cost, self.optimizer], feed_dict=feed_dict)
                epoch_loss += minibatch_cost
            epoch_loss /= num_batches
            if verbose:
                print("Loss for epoch {}: {}".format(epoch, epoch_loss))

    def predict(self, X_test):
        """
        Make predictions using the trained model.

        :param X_test: array; test data; shape should be [n_samples, sequence_length]
        :return: array; the predicted labels
        """

        predicted = tf.nn.sigmoid(self.logits)
        pred = self.sess.run(predicted, feed_dict={self.X: X_test, self.keep_prob: 1})
        pred = np.where(pred > 0.5, 1, 0)
        return pred

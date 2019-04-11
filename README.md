# rnn_lstm

LSTM models for binary text classification in tensorflow and keras.


### Usage

##### Data

The imdb dataset can be downloaded and formatted using rnn_lstm.utils.data_utils.get_data()

```python
from rnn_lstm.model.data_utils import get_data, get_minibatches
X_train, X_dev, y_train, y_dev = get_data()

# For the tensorflow model, preprocess into batches
batches = get_minibatches(X_train, y_train)
```

##### Training

The models can be fit as follows:

```python
# Keras model
from rnn_lstm.model.keras_model import KerasLSTM
lstm = KerasLSTM()
lstm.fit(X_train, y_train)

# Tensorflow model
from rnn_lstm.model.tensorflow_model import TfLSTM
lstm = TfLSTM(TfLSTM(5000, 2))
lstm.fit(batches)
```

##### Assessment

The models can be assessed in the following way:

```python
from sklearn.metrics import f1_score
train_predicted = lstm.predict(X_train)
test_predicted = lstm.predict(X_dev)
print('Train set f1: {}'.format(f1_score(train_predicted, y_train)))
print('Test set f1: {}'.format(f1_score(test_predicted, y_test)))
```


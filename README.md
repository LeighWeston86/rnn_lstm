# rnn_lstm

LSTM models for binary text classification in tensorflow and keras. The models have a simple and high-level interface for training and assessment.


### Usage

##### Data

The imdb dataset can be downloaded and formatted using the rnn_lstm.utils.data_utils.get_data function (set y_as_column=True for tensorflow model).

```python
from rnn_lstm.model.data_utils import get_data
X_train, X_test, y_train, y_test = get_data()
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
lstm = TfLSTM(TfLSTM(5000, 1))
lstm.fit(X_train, y_train)
```

##### Prediction and assessment

The models can be assessed in the following way:

```python
rnn_lstm.model.data_utils import assess
predicted = lstm.predict(X_test)
scores = assess(predicted, y_test)
print(scores["f1"])
```


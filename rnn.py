import tensorflow as tf

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

TRAIN_SPLIT = 70
df_past_history = 15
df_future_target = 0
BATCH_SIZE = 256
BUFFER_SIZE = 2
EVALUATION_INTERVAL = 200
EPOCHS = 10


def get_data(dataset, start_index, end_index, history_size, target_size):
  data = []
  labels = []

  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size

  for i in range(start_index, end_index):
    indices = range(i-history_size, i)
    data.append(np.reshape(dataset[indices], (history_size, 1)))
    labels.append(dataset[i+target_size])
  return np.array(data), np.array(labels)


def getPredict(data):
    df = pd.DataFrame({'Parameter': data[:, 0]})
    # стандартизация. нужна ли?
    df_train_mean = df[:TRAIN_SPLIT].mean()
    df_train_std = df[:TRAIN_SPLIT].std()
    df = (df - df_train_mean) / df_train_std

    x_train, y_train = get_data(df, 0, TRAIN_SPLIT,
                                df_past_history,
                                df_future_target)
    x_val, y_val = get_data(df, TRAIN_SPLIT, None,
                            df_past_history,
                            df_future_target)

    train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

    val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val = val.batch(BATCH_SIZE).repeat()

    simple_lstm_model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(8, input_shape=x_train.shape[-2:]),
        tf.keras.layers.Dense(1)
    ])
    simple_lstm_model.compile(optimizer='adam', loss='mae')
    simple_lstm_model.fit(train, epochs=EPOCHS,
                          steps_per_epoch=EVALUATION_INTERVAL,
                          validation_data=val, validation_steps=50)

    x,y = val.take(1)
    return simple_lstm_model.predict(x)
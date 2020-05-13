# -*- coding: utf-8 -*-
"""
Created on Mon May 11 11:51:17 2020

@author: matte
"""

import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, GlobalMaxPooling1D, SpatialDropout1D, Input
from utils import generate_fake_data

batch_size = 512
epochs = 8


model_lstm = Sequential()
#model_lstm.add(Embedding(10, 512, input_length=10))
model_lstm.add(LSTM(10))
model_lstm.add(Dense(2, activation = 'softmax'))
model_lstm.compile(
    loss='categorical_crossentropy',
    optimizer='Adam',
    metrics=['accuracy']
)

X_train, y_train = generate_fake_data()
y_train = to_categorical(y_train, 2)
history = model_lstm.fit(
    X_train,
    y_train,
    epochs = 8,
    batch_size = 1
)

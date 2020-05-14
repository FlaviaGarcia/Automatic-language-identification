# -*- coding: utf-8 -*-
"""
Created on Thu May 14 11:41:06 2020

@author: FlaviaGV, MatteoDM, CarlesBR, TheodorosPP
"""

import sys
sys.path.append("..")

from tensorflow.keras.utils import to_categorical
from models import utils
from models.LSTM import LSTM


n_frames_utterance = 10 

fake_features, fake_targets = utils.generate_fake_data(n_utterances=10,
                                                       n_frames_utterance=n_frames_utterance)
fake_targets_categ = to_categorical(fake_targets)

x_train = fake_features[:70]
y_train = fake_targets_categ[:70]
x_val = fake_features[70:80]
y_val = fake_targets_categ[70:80]
x_test = fake_features[80:]
y_test = fake_targets_categ[80:]

n_features = fake_features.shape[1]
n_output_nodes=fake_targets_categ.shape[1]


n_memory_cells = 512
batch_size=10 
n_epochs=20 
dropout_rate_input=0.0 
dropout_rate_hidden=0.0 
optimizer_method="adam"

lstm = LSTM(n_memory_cells, n_output_nodes, 
            n_frames_utterance, n_features, dropout_rate_input, 
            dropout_rate_hidden, optimizer_method)

lstm.train(x_train, y_train, x_val, y_val, batch_size, n_epochs)

scores = lstm.predict_proba(x_test)

classes = lstm.predict_classes(x_test)

"""

n_frames_utterance = 10 
n_utterances=10
fake_features, fake_targets = utils.generate_fake_data(n_utterances,
                                                       n_frames_utterance=n_frames_utterance)

n_features = fake_features.shape[1]

fake_features = fake_features.reshape(n_utterances, n_frames_utterance, n_features)


fake_targets_categ = to_categorical(fake_targets)

fake_targets_categ = fake_targets_categ.reshape(n_utterances, n_frames_utterance, -1)


X_train = fake_features[:7]
y_train = fake_targets_categ[:7]
X_test = fake_features[7:]
y_test = fake_targets_categ[7:]


#y_train = y_train.reshape(y_train.shape[0], 1, y_train.shape[1])
#y_test = y_test.reshape(y_test.shape[0], 1, y_test.shape[1])

model = Sequential()
model.add(LSTM(units=32, kernel_initializer='uniform',
           unit_forget_bias='one', activation='tanh', recurrent_activation='sigmoid', 
           return_sequences=True))

nb_classes = fake_targets_categ.shape[2]
model.add(Dense(nb_classes, activation="softmax"))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
#print(model.summary())

print("Train...")
model.fit(X_train, y_train, batch_size=10, epochs=3)


classes = model.predict_classes(X_test)
proba = model.predict_proba(X_test)



"""

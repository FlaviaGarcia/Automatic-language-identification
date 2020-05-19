# -*- coding: utf-8 -*-
"""
Created on Thu May 14 11:41:06 2020

@author: FlaviaGV, MatteoDM, CarlesBR, TheodorosPP
"""

import sys
sys.path.append("..")

from tensorflow.keras.utils import to_categorical
from models import utils
from models.LSTMSpeech import LSTMSpeech

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
dropout_rate_input=0.2
dropout_rate_hidden=0.3 
optimizer_method="adam"

lstm = LSTMSpeech(n_memory_cells, n_output_nodes, 
                  n_frames_utterance, n_features, dropout_rate_input, 
                  dropout_rate_hidden, optimizer_method)

lstm.train(x_train, y_train, x_val, y_val, batch_size, n_epochs)

scores = lstm.predict_proba(x_test)

classes = lstm.predict_classes(x_test)




import sys
sys.path.append("..")

from tensorflow.keras.utils import to_categorical
from models import utils
from models.CNNSpectMFCC import CNNSpectMFCC
import numpy as np

n_frames_utterance = 21
fake_features, fake_targets = utils.generate_fake_data(n_utterances=10,
                                                       n_frames_utterance=n_frames_utterance, n_features=50)
fake_targets_categ = to_categorical(fake_targets)

fake_features = fake_features.reshape(-1, 21, 50, 1)

input_dims = [(21, 50, 1), (21, 50, 1)]
pool_size_2d = 5
divide_pool_size_1d = 4
filters_size = [5, 5]
n_filters = [128, 256]
model = CNNSpectMFCC(input_dims, n_filters, filters_size, pool_size_2d, divide_pool_size_1d, n_output_nodes=2)

model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])

model.fit([fake_features, fake_features], fake_targets_categ[np.random.randint(0, 100, 10)], epochs=10, batch_size=10)

#model.summary()
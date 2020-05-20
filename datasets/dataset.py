import keras
import os
from sidekit.frontend.features import compute_delta, plp
import librosa
import sidekit
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio

class DataGenerator(keras.utils.Sequence):

    def __init__(self, root_path, batch_size=32, shuffle=True):

        self.batch_size = batch_size
        self.root_path = root_path
        self.shuffle = shuffle
        self.on_epoch_end()
        self.list_dir = []
        self.dim = (128, 79)
        self.n_channels = 1
        targets = os.listdir(self.root_path)
        self.n_classes
        self.target_to_class = {lang: i for i, lang in enumerate(targets)}
        i = 0
        for target in target:
            target_path = os.path.join(os.path.join(self.root_path, 'clips_cut'), target)
            for path in os.listdir(target_path):
                item_path = os.path.join(target_path, path)
                self.list_dir.append(item_path)
                
            i += 1

        self.n_classes = len(targets)



    def __len__(self):

        return len(self.list_dir)

    def __getitem__(self, index):
        
        raise Exception('Not implemented')
        # Generate indexes of the batch
        indexes = self.indexes[index]
        list_dirs_temp = [self.list_dir[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(indexesm list_dirs_temp)

        return X, y

    def on_epoch_end(self):

        self.indexes = np.arange(len(self.list_dir))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_dirs_temp):

        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, item_path in enumerate(list_IDs_temp):
            # Store sample
            
            audio_binary = tf.io.read_file(item_path)
            waveform = tfio.audio.decode_mp3(audio_binary).numpy().reshape(-1)
            mel = librosa.feature.melspectrogram(y, sr=16000)
            ps_db = librosa.power_to_db(mel, ref=np.max).reshape((*self.dim, 1))
            X[i,] = ps_db
            # Store class
            label = item_path.split('/')[-3]
            y[i] = self.target_to_class[label]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)


"""

model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    workers=6)

"""
import keras
import os
from sidekit.frontend.features import compute_delta, plp
import librosa
import sidekit

class DataGenerator(keras.utils.Sequence):

    def __init__(self, root_path, batch_size=32, shuffle=True, dynamic=False):

        self.batch_size = batch_size
        self.root_path = root_path
        self.shuffle = shuffle
        self.on_epoch_end()
        self.list_dir = []
        self.dynamic = dynamic
        targets = os.listdir(self.root_path)
        self.n_classes
        target_to_class = {lang: i for i, lang in enumerate(targets)}
        i = 0
        for target in target:
            target_path = os.path.join(self.root_path, target)
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

        # Generate data
        X, y = self.__data_generation(indexes, dynamic)

        return X, y

    def on_epoch_end(self):

        self.indexes = np.arange(len(self.list_dir))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes, dynamic):

        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load('data/' + ID + '.npy')

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)


"""

model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    workers=6)

"""
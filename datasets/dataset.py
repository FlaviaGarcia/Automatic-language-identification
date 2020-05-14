import keras
import os

class DataGenerator(keras.utils.Sequence):

    def __init__(self, root_path, batch_size=32, shuffle=True):

        self.batch_size = batch_size
        self.root_path = root_path
        self.shuffle = shuffle
        self.on_epoch_end()
        self.list_dir = []
        self.target_dict = {}
        self.item_to_target = {}
        targets = os.listdir(self.root_path)
        self.n_classes
        i = 0
        for target in target:
            target_path = os.path.join(self.root_path, target)
            self.target_dict[i] = target
            for path in os.listdir(target_path):
                item_path = os.path.join(target_path, path)
                self.list_dir.append(item_path)
                
            i += 1

        self.n_classes = i



    def __len__(self):

        return int(np.floor(len(self.list_dir) / self.batch_size))

    def __getitem__(self, index):
        
        raise Exception('Not implemented')
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):

        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):

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
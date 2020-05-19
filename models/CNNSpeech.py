from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, MaxPooling1D



class CNNSpeech:

    def __init__(self, input_dims, n_filters, filters_size, pool_size_2d, divide_pool_size_1d,
                 n_output_nodes, batch_normalization=True):

        self.input_dims = input_dims
        self.n_filters = n_filters
        self.filters_size = filters_size
        self.pool_size_2d = pool_size_2d
        self.divide_pool_size_1d = divide_pool_size_1d

        self.batch_normalization = batch_normalization
        self.n_output_nodes = n_output_nodes

        self.activation_func = 'relu'
        self.output_activation_func = 'softmax'
        self.optimizer_method = 'adam'
        self.loss = 'categorical_crossentropy'
        """"
        self.dropout_rate = dropout_rate
        """
        self.create_NN()


    def create_NN(self):
        self.model = Sequential()

        self.model.add(Conv2D(filters=self.n_filters[0],
                              kernel_size=self.filters_size[0],
                              activation='relu',
                              use_bias=True,
                              strides=1,
                              padding='same',
                              input_shape=self.input_dims))

        if self.batch_normalization:
            self.model.add(BatchNormalization())

        self.model.add(MaxPooling2D(pool_size=self.pool_size_2d,
                                    strides=None,
                                    padding='same'))

        self.model.add(Conv2D(filters=self.n_filters[1],
                              kernel_size=self.filters_size[1],
                              activation='relu',
                              use_bias=True,
                              strides=1,
                              padding='same'))

        if self.batch_normalization:
            self.model.add(BatchNormalization())

        prev_layer_shape = self.model.layers[-1].output_shape

        if prev_layer_shape[1] % self.divide_pool_size_1d != 0:
            raise ValueError("Frequencies are not divisible by " + self.divide_pool_size_1d)

        pool_size_1d = prev_layer_shape[1]//self.divide_pool_size_1d

        self.model.add(MaxPooling2D(pool_size=(pool_size_1d, prev_layer_shape[2])))

        self.model.add(Flatten())

        self.model.add(Dense(self.n_output_nodes, activation=self.output_activation_func))

        self.model.compile(loss=self.loss,
                           optimizer=self.optimizer_method,
                           metrics=['accuracy'])

    def train(self, features_train, targets_train, features_val, targets_val,
              batch_size, n_epochs, verbose=1):

        features_train, targets_train = self.convert_data(features_train, targets_train)
        features_val, targets_val = self.convert_data(features_val, targets_val)

        model_info = self.model.fit(features_train, targets_train, batch_size=batch_size,
                                    validation_data=(features_val, targets_val),
                                    verbose=verbose, epochs=n_epochs)

        return model_info

    def predict_classes(self):
        """ Majority vote of the classes """
        pass


"""
input_dims = (20,40,1)
pool_size_2d = 5
divide_pool_size_1d = 4
filters_size = [5, 5]
n_filters = [128, 256]
cnn = CNNSpeech(input_dims, n_filters, filters_size, pool_size_2d, divide_pool_size_1d, n_output_nodes=2)

cnn.model.summary()
"""
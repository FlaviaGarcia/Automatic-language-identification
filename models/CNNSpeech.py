from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, MaxPooling1D


def create_CNN(input_dims, n_filters, filters_size, pool_size_2d, divide_pool_size_1d,
               n_output_nodes, batch_normalization=True):
    model = Sequential()

    model.add(Conv2D(filters=n_filters[0],
                     kernel_size=filters_size[0],
                     activation='relu',
                     use_bias=True,
                     strides=1,
                     padding='same',
                     input_shape=input_dims))

    if batch_normalization:
        model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=pool_size_2d,
                           strides=None,
                           padding='same'))

    model.add(Conv2D(filters=n_filters[1],
                     kernel_size=filters_size[1],
                     activation='relu',
                     use_bias=True,
                     strides=1,
                     padding='same'))

    if batch_normalization:
        model.add(BatchNormalization())

    prev_layer_shape = model.layers[-1].output_shape

    pool_size_1d = prev_layer_shape[1] // divide_pool_size_1d

    model.add(MaxPooling2D(pool_size=(pool_size_1d, prev_layer_shape[2])))

    model.add(Flatten())

    model.add(Dense(n_output_nodes, activation='softmax'))

    return model


"""
model.compile(loss=self.loss,
                           optimizer=self.optimizer_method,
                           metrics=['accuracy'])
input_dims = (x,x,1)
pool_size_2d = 5
divide_pool_size_1d = 4
filters_size = [5, 5]
n_filters = [128, 256]
cnn = CNNSpeech(input_dims, n_filters, filters_size, pool_size_2d, divide_pool_size_1d, n_output_nodes=2)

cnn.model.summary()
"""

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten


class CNNSpectMFCC(tf.keras.Model):

    def __init__(self, input_dims, n_filters, filters_size, pool_size_2d, divide_pool_size_1d,
                 n_output_nodes, batch_normalization=True):
        super(CNNSpectMFCC, self).__init__()
        self.input_dims = input_dims
        self.n_filters = n_filters
        self.filters_size = filters_size
        self.pool_size_2d = pool_size_2d
        self.divide_pool_size_1d = divide_pool_size_1d

        self.batch_normalization = batch_normalization
        self.n_output_nodes = n_output_nodes

        self.conv_step1_1 = Conv2D(filters=self.n_filters[0],
                                   kernel_size=self.filters_size[0],
                                   activation='relu',
                                   use_bias=True,
                                   strides=1,
                                   padding='same',
                                   input_shape=self.input_dims[0])
        self.conv_step1_2 = Conv2D(filters=self.n_filters[0],
                                   kernel_size=self.filters_size[0],
                                   activation='relu',
                                   use_bias=True,
                                   strides=1,
                                   padding='same',
                                   input_shape=self.input_dims[1])
        self.batch_norm_step1_1 = BatchNormalization()
        self.batch_norm_step1_2 = BatchNormalization()
        self.max_pooling_step1_1 = MaxPooling2D(pool_size=self.pool_size_2d,
                                                strides=None,
                                                padding='same')
        self.max_pooling_step1_2 = MaxPooling2D(pool_size=self.pool_size_2d,
                                                strides=None,
                                                padding='same')
        self.conv_step2_1 = Conv2D(filters=self.n_filters[1],
                                   kernel_size=self.filters_size[1],
                                   activation='relu',
                                   use_bias=True,
                                   strides=1,
                                   padding='same')
        self.conv_step2_2 = Conv2D(filters=self.n_filters[1],
                                   kernel_size=self.filters_size[1],
                                   activation='relu',
                                   use_bias=True,
                                   strides=1,
                                   padding='same')
        self.batch_norm_step2_1 = BatchNormalization()
        self.batch_norm_step2_2 = BatchNormalization()

        self.dense1 = tf.keras.layers.Dense(200, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(200, activation=tf.nn.relu)
        self.fc_layer = tf.keras.layers.Dense(400, activation=tf.nn.relu)
        self.output_layer = tf.keras.layers.Dense(n_output_nodes, activation=tf.nn.softmax)

    def call(self, inputs, training=False):
        mfcc_input = inputs[0]
        spectogram_input = inputs[1]
        x1 = self.conv_step1_1(mfcc_input)
        x2 = self.conv_step1_2(spectogram_input)
        if self.batch_normalization:
            x1 = self.batch_norm_step1_1(x1)
            x2 = self.batch_norm_step1_2(x2)
        x1 = MaxPooling2D(pool_size=self.pool_size_2d,
                          strides=None,
                          padding='same')(x1)
        x2 = MaxPooling2D(pool_size=self.pool_size_2d,
                          strides=None,
                          padding='same')(x2)
        x1 = self.conv_step2_1(x1)
        x2 = self.conv_step2_2(x2)
        if self.batch_normalization:
            x1 = self.batch_norm_step2_1(x1)
            x2 = self.batch_norm_step2_2(x2)


        prev_layer_shape_1 = x1.shape

        prev_layer_shape_2 = x1.shape

        pool_size_1d_1 = prev_layer_shape_1[1] // self.divide_pool_size_1d
        pool_size_1d_2 = prev_layer_shape_2[1] // self.divide_pool_size_1d

        print("#####", pool_size_1d_1)
        print("#####", pool_size_1d_2)


        x1 = MaxPooling2D(pool_size=(pool_size_1d_1, prev_layer_shape_1[2]))(x1)
        x2 = MaxPooling2D(pool_size=(pool_size_1d_2, prev_layer_shape_2[2]))(x2)

        x1 = Flatten()(x1)
        x2 = Flatten()(x2)
        x1 = self.dense1(x1)
        x2 = self.dense1(x2)
        x = tf.keras.layers.concatenate([x1, x2])
        x = self.fc_layer(x)

        return self.output_layer(x)

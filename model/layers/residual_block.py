import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, Lambda, Add

class ResidualBlock(Layer):
    def __init__(self, name, n_filters=64, k_size=3, scale=2, **kwargs):
        super(ResidualBlock, self).__init__(name=name, **kwargs)

        self.n_filters = n_filters
        self.k_size= k_size
        self.scale = scale

        self.conv1 = Conv2D(self.n_filters, self.k_size, padding='same', activation='relu', name='rb_conv_1')
        self.conv2 = Conv2D(self.n_filters, self.k_size, padding='same', name='rb_conv_2')

    
    def call(self, inputs, **kwargs):
        layer = self.conv1(inputs)
        layer = self.conv2(layer)
        layer = Lambda(lambda t: t * self.scale)(layer)

        return Add()([inputs, layer])

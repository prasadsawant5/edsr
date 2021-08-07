import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Add, Lambda
from model.layers.residual_block import ResidualBlock
from utils import pixel_shuffle


class MyModel:
    def __init__(self) -> None:
        self.n_res_block = 8
        self.num_filters = 64
        self.k_size = 3
        self.scale = 2
    

    def build(self, x: tf.keras.Input) -> Model:
        conv_1 = Conv2D(self.num_filters, self.k_size, padding='same', name='conv_1')(x)

        res_block = None
        for i in range(self.n_res_block):
            with tf.name_scope('res_block_{}'.format(i + 1)):
                residual_block = ResidualBlock('res_block_{}'.format(i + 1))
                if i == 0:
                    res_block = residual_block(conv_1)
                else:
                    res_block = residual_block(res_block)

        conv_final = Conv2D(self.num_filters, self.k_size, padding='same', name='conv_final')(res_block)
        conv_final = Add(name='add_1')([conv_1, conv_final])

        with tf.name_scope('upsampling'):
            upsample_conv_1 = Conv2D(self.num_filters * (self.scale ** 2), kernel_size=self.k_size, padding='same', name='upsample_conv_1')(conv_final)
            # pix_shuffle = Lambda(pixel_shuffle(self.scale))(upsample_conv_1)
            pix_shuffle = tf.nn.depth_to_space(upsample_conv_1, self.scale, name='pixel_shuffle')

        output = Conv2D(3, self.k_size, padding='same', name='output')(pix_shuffle)

        return Model(x, output, name='edsr')
